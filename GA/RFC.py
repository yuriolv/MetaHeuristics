import random as rd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd

class GA:
    def __init__(self, generations, num_individuals):
        self.generations = generations
        self.num_individuals = num_individuals
        self.mutation_rate = 15
        self.population = []

    def generate_chromosome(self, individual):
        for j in range(5):
                if(j == 0):
                    individual.chromosome.append(rd.randint(1, 3))
                elif(j == 1):
                    individual.chromosome.append(rd.randint(10, 100))
                elif(j == 2):
                    individual.chromosome.append(rd.randint(3, 70))
                elif(j == 3):
                    individual.chromosome.append(rd.randint(2, 70))
                elif(j == 4):
                    individual.chromosome.append(rd.randint(1, 2))

    def chromosome_encoding(self, individual):
        coded = list(individual.chromosome)

        if(individual.chromosome[0] == 1):
            coded[0] = 'gini'
        elif(individual.chromosome[0] == 2):
            coded[0] = 'entropy'
        elif(individual.chromosome[0] == 3):
            coded[0] = 'log_loss'

        if(coded[4] == 1):
            coded[4] = 'sqrt'
        elif(coded[4] == 2):
            coded[4] = 'log2'

        return coded

    def generate_population(self):
        for i in range(self.num_individuals):
            i = Individual()

            self.generate_chromosome(i)
        
            self.population.append(i)

    def evaluate_individual(self, individual):
        coded_chromosome = self.chromosome_encoding(individual)
        df = pd.read_csv('new_base.csv')
        y = df['RainTomorrow']
        x = df.drop(columns=['RainTomorrow'])   

        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(
                    criterion=coded_chromosome[0], 
                    n_estimators=coded_chromosome[1], 
                    max_depth=coded_chromosome[2],
                    min_samples_split=coded_chromosome[3],
                    max_features=coded_chromosome[4]
                    )
        model.fit(x_treino, y_treino)

        y_predicoes = model.predict(x_teste)

        return accuracy_score(y_teste, y_predicoes)

    def evaluate_population(self):
        for i in tqdm(self.population):
            i.accuracy = round(self.evaluate_individual(i), 3)

    def tournament(self):
        ind1, ind2 = rd.sample(self.population, 2)
        if(ind1.accuracy > ind2.accuracy):
            return ind1
        else:
            return ind2
        
    def crossover(self, dad1, dad2, index):
        new_ind = Individual()

        new_ind.chromosome = dad2.chromosome[:index]
        new_ind.chromosome.extend(dad1.chromosome[index:])

        return new_ind

    def update_population(self):
        new_pop = []
        for j in range(5):
            index = rd.randint(1, 4)
            dad1, dad2 = self.tournament(), self.tournament()

            new_ind1, new_ind2 = self.crossover(dad1, dad2, index), self.crossover(dad2, dad1,index)

            new_pop.extend([new_ind1, new_ind2])

        self.population = new_pop 

    def mutation(self):
        for i in self.population:
            for j in range(5):     
                probability = rd.randint(1,100)
                if(probability <= self.mutation_rate):
                    if(j == 0):
                        i.chromosome[j] = rd.randint(1, 3)
                    elif(j == 1):
                        i.chromosome[j] = rd.randint(10, 100)
                    elif(j == 2):
                        i.chromosome[j] = rd.randint(3, 70)
                    elif(j == 3):
                        i.chromosome[j] = rd.randint(2, 70)
                    elif(j == 4):
                        i.chromosome[j] = rd.randint(1, 2)

    def show_population(self):
        print("\tAccuracy\t|\tChromosome\t")
        for i in self.population:
            print('\t',i.accuracy,'\t','\t    ', i.chromosome)
        print("\n\n")

class Individual:
    def __init__(self):
        self.chromosome = []
        self.accuracy = 0 

def initialize(num_individuals, generations):
        ga = GA(generations, num_individuals)
        ga.generate_population()
        ga.evaluate_population()
        print('Generetion 1')
        ga.show_population()
        for i in range(ga.generations - 1):
            ga.update_population()
            ga.mutation()
            ga.evaluate_population()
            print("Generation ", i+2)
            ga.show_population()

initialize(10, 10)
