import random as rd

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
        individual.chromosome.append(rd.choice(['adam', 'sgd', 'rmsprop']))
        individual.chromosome.append(round(rd.random(), 2))
        multiples_of_2 = [i for i in range(2, 100 + 1) if i % 2 == 0]
        layers_size = rd.choice(multiples_of_2)
        for j in range(int(layers_size/2)):
            individual.chromosome.append(rd.randint(1,30)) 
            individual.chromosome.append(rd.choice(['relu', 'sigmoid', 'tanh']))

    def generate_population(self):
        for i in range(self.num_individuals):
            i = Individual()

            self.generate_chromosome(i)
        
            self.population.append(i)

    def evaluate_individual(self, individual):
        df = pd.read_csv('new_base.csv')
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
        for i in range(2, len(individual.chromosome)- 2):
            if i % 2 == 0:
                model.add(Dense(individual.chromosome[i], activation=individual.chromosome[i+1]))
                model.add(Dropout(individual.chromosome[1]))
        model.add(Dense(2, activation='softmax'))

        model.compile(
                        optimizer=individual.chromosome[0], loss='binary_crossentropy', 
                        metrics=['accuracy']
                        )

        model.fit(x_treino, y_treino, epochs=100, verbose=1, batch_size=32)

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
        for j in range(len(self.population)/2):
            dad1, dad2 = self.tournament(), self.tournament()
            index = rd.randint(1, min(len(dad1.chromosome), len(dad2.chromosome)))

            new_ind1, new_ind2 = self.crossover(dad1, dad2, index), self.crossover(dad2, dad1,index)

            new_pop.extend([new_ind1, new_ind2])

        self.population = new_pop 

    def mutation(self):
        for i in self.population:
            probability = rd.randint(1,100)
            if(probability <= self.mutation_rate):
                i.chromosome = []     
                i.chromosome.append(rd.choice(['adam', 'sgd', 'rmsprop']))
                i.chromosome.append(round(rd.random(), 2))

                multiples_of_2 = [i for i in range(2, 100 + 1) if i % 2 == 0]
                layers_size = rd.choice(multiples_of_2)
                for j in range(int(layers_size/2)):
                    i.chromosome.append(np.randint(1,30))
                    i.chromosome.append(np.choice(['relu', 'sigmoid', 'tanh']))

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
