import random as rd
from sklearn import ml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import numpy as np

from tensorflow import keras
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.utils import to_categorical
from keras.src.optimizers import Adam

class GA:
    def __init__(self, generations, num_individuals):
        self.generations = generations
        self.num_individuals = num_individuals
        self.mutation_rate = 15
        self.population = []

    def generate_chromosome(self, individual):
        for j in range(6):
                if(j == 0):#numero de camadas
                    individual.chromosome.append(rd.randint(1, 10))
                elif(j == 1):#numero de neuronios em cada camada
                    individual.chromosome.append(np.randint(1,30, individual.chromosome[0]))
                elif(j == 2):#função de ativação para cada camada
                    individual.chromosome.append(np.randint(1, 3, individual.chromosome[0]))
                elif(j == 3):#taxa de aprendizado
                    float_dot = rd.choice([10,100,1000])
                    individual.chromosome.append(1/float_dot)
                elif(j == 4):#epocas
                    individual.chromosome.append(rd.randint(1, 100))
                elif(j == 5):#dropout
                    individual.chromosome.append(rd.ramdom())

    def chromosome_encoding(self, individual):
        coded = list(individual.chromosome)
        for i in range(len(coded[2])):
            if(individual.chromosome[2, i] == 1):
                coded[2, i] = 'relu'
            elif(individual.chromosome[2, i] == 2):
                coded[2, i] = 'sigmoid'
            elif(individual.chromosome[2, i] == 3):
                coded[2, i] = 'tanh'

        return coded

    def generate_population(self):
        for i in range(self.num_individuals):
            i = Individual()

            self.generate_chromosome(i)
        
            self.population.append(i)

    def evaluate_individual(self, individual):
        coded_chromosome = self.chromosome_encoding(individual)
        df = pd.read_csv('../new_base.csv')
        y = df['RainTomorrow']
        x = df.drop(columns=['RainTomorrow']) 

        x_treino, x_teste, y_treino, y_teste = train_test_split(
                                                                    x, y, test_size=0.2, 
                                                                    random_state=42
                                                                    )
        y_treino = to_categorical(y_treino)
        y_teste = to_categorical(y_teste)

        model = Sequential() #número de camadas
        model.add(keras.Input(shape=(25,)))
        for i in range(len(individual[0])):
            model.add(Dense(individual[1, i]), activation=individual[2, i])
            model.add(Dropout(individual[5]))

        optmizer = Adam(learning_rate=individual[3])

        model.compile(
                        optimizer='adam', loss='binary_crossentropy', 
                        metrics=['accuracy'], verbose=1
                        )

        model.fit(x_treino, y_treino, epochs=individual[4])

        perca, acuracia = model.evaluate(x_teste, y_teste, verbose=0)

        return acuracia

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
            index = rd.randint(1, 5)
            dad1, dad2 = self.tournament(), self.tournament()

            new_ind1, new_ind2 = self.crossover(dad1, dad2, index), self.crossover(dad2, dad1,index)

            new_pop.extend([new_ind1, new_ind2])

        self.population = new_pop 

    def mutation(self):
        for i in self.population:
            for j in range(6):     
                probability = rd.randint(1,100)
                if(probability <= self.mutation_rate):
                    if(j == 0):#numero de camadas
                        i.chromosome.append(rd.randint(1, 10))
                    elif(j == 1):#numero de neuronios em cada camada
                        i.chromosome.append(np.randint(1,30, i.chromosome[0]))
                    elif(j == 2):#função de ativação para cada camada
                        i.chromosome.append(np.randint(1, 3, i.chromosome[0]))
                    elif(j == 3):#taxa de aprendizado
                        float_dot = rd.choice([10,100,1000])
                        i.chromosome.append(1/float_dot)
                    elif(j == 4):#epocas
                        i.chromosome.append(rd.randint(1, 100))
                    elif(j == 5):#dropout
                        i.chromosome.append(rd.ramdom())

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
