import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from keras.src.models import Sequential
from keras.src.layers import Dropout, Dense
from numpy.random import randint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from random import choice
from random import uniform


class StructuredDataPreprocessing:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.numerical_features = list(X.select_dtypes(include=[np.int64, np.float64]).columns)
        self.categorical_features = list(X.select_dtypes(include=[np.object_]).columns)
    
    def featuresEncoder(self) -> pd.DataFrame:
        """
        Encode categorical features as a numerical column.

        Parameters
        ----------
        X: DataFrame
            Features data.

        Returns
        -------
            DataFrame
        """
        le = LabelEncoder()

        for name_column in self.numerical_features:
            self.X[name_column] = self.X[name_column].astype(float)

        
        for name_column in self.categorical_features:
            self.X[name_column] = le.fit_transform(self.X[name_column])
            if len(self.X[name_column].unique()) >= self.X.shape[0]/(3/2):
                self.X = self.X.drop(columns=[name_column])

        pd.DataFrame(self.X).to_numpy()

        return self.X

    def labelncoder(self) -> pd.DataFrame:
        """
        Encode target labels. The values ​​used will be between 0 and (n_classes - 1).

        Parameters
        ----------
        y: DataFrame
            Target labels data.

        Returns
        -------
            DataFrame
        """

        if str(self.y.dtype) == 'object':
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)
            pd.DataFrame(self.y).to_numpy()
        else:
            pd.DataFrame(self.y).to_numpy()
            if min(self.y.unique()) > 0:
                for i in range(len(self.y)):
                    self.y[i] -= 1
        

        return self.y

test_size = 0.3

mHL = 5
limitations = [
    (mHL*3)+2,          # vector size

    [0, 1],             # optimizers (adam, sgd)
    range(40, 151),     # epochs
    
    range(2, mHL+1),    # number of hiddenlayers

    range(3, 501),      # neurons per layer
    [0, 1],             # activation functions (relu, elu)
    [0, 0.3],           # dropouts 
]

# Example of vector:
# vec = [optimizer, epochs, num_neur[0], activation[0], dropout[0], num_neur[1], activation[1], dropout[1], ...]

def calculate_distance(initial, final):
    resultant = []
    for j in range(len(final)):
        resultant.append(round(final[j] - initial[j], 4))
    
    return resultant

def sum_vectors(vector_list):
    resultant = []
    for i in range(len(vector_list[0])):
        component = 0
        for vector in vector_list:
            component += vector[i]

        resultant.append(round(component, 4))

    return kill_layers(limit_values(resultant))

def calculate_velocity(vector_list):
    resultant = []
    for i in range(len(vector_list[0])):
        component = 0
        for vector in vector_list:
            component += vector[i]
        
        if i == 1:
            if component > int(max(limitations[2])/4):
                component = int(max(limitations[2])/4)
            elif component < (int(max(limitations[2])/4)) * (-1):
                component = (int(max(limitations[2])/4)) * (-1)
        elif i >= 2 and i%3 == 2:
            if component > int(max(limitations[4])/4):
                component = int(max(limitations[4])/4)
            elif component < (int(max(limitations[4])/4)) * (-1):
                component = (int(max(limitations[4])/4)) * (-1)
        
        resultant.append(round(component, 4))

    return resultant
        
def kill_layers(vet):
    count = 0
    for layer in range(len(vet)-1, 2, -3):
        if not (vet[layer] and vet[layer-2]):
            count += 1
            vet.pop(layer)
            vet.pop(layer-1)
            vet.pop(layer-2)
    
    for i in range(count*3):
      vet.append(0)
    return vet

def limit_values(vet):
    last_optimizer = max(limitations[1])
    if vet[0] > last_optimizer:
        vet[0] = last_optimizer
    elif vet[0] < 0:
        vet[0] = 0
    
    max_epochs = max(limitations[2])
    min_epochs = min(limitations[2])
    if vet[1] > max_epochs:
        vet[1] = max_epochs
    elif vet[1] < min(limitations[2]):
        vet[1] = min_epochs

    for layer in range(2, len(vet), 3):
        max_neurons = max(limitations[4])
        min_neurons = min(limitations[4])
        if vet[layer] > max_neurons:
            vet[layer] = max_neurons        
        elif vet[layer] < min_neurons:
            vet[layer] = 0 # kill layer


        for i in range(1,3):
            if vet[layer+i] < 0:
                vet[layer+i] = 0
        
        last_activation = max(limitations[5])
        if vet[layer+1] > last_activation:
            vet[layer+1] = last_activation

        max_dropout = max(limitations[6])
        if vet[layer+2] > max_dropout:
            vet[layer+2] = max_dropout
        
    return vet   

def scalar(vet, a):
    for i in range(len(vet)):
        if i > 2 and i%3 == 1:
            vet[i] = round((vet[i] * a), 4)
        else:
            vet[i] = int((vet[i] * a))
    return vet

def multiply_vectors(vet1, vet2):
    new_vet = []
    for i in range(len(vet1)):
        if i > 2 and i%3 == 1:
            new_vet.append(round(vet1[i] * vet2[i], 4))
        else:
            new_vet.append(int(vet1[i] * vet2[i]))
    
    return new_vet

def init_hiperparams(number_of_classes):
        global output_size, loss_func, output_activation
        output_size = 1
        output_activation = "sigmoid"
        loss_func = "binary_crossentropy"
        if number_of_classes > 2:
            output_size = number_of_classes
            loss_func = "sparse_categorical_crossentropy"
            output_activation = "softmax"

class Particle:
    def __init__(self, current_vel, current_position, personal_best, c1, c2):
        self.current_vel = current_vel
        self.current_position = current_position
        self.personal_best = personal_best
        self.c1 = c1
        self.c2 = c2
        self.best_performance = 0

    def __repr__(self):
        return "" + str(self.current_vel) + str(self.current_position) + str(self.personal_best) + str(self.c1) + str(self.c2)

    def next_velocity(self, w, global_best, vec_size):
        inertia = scalar(self.current_vel, w)

        personal_best_dist = calculate_distance(self.current_position, self.personal_best)
        random_vel = [round(uniform(0, 1), 4) for _ in range(vec_size)]
        personal_vel = multiply_vectors(personal_best_dist, random_vel)
        personal_comp = scalar(personal_vel, self.c1)
        print("Personal Best: " + str(self.personal_best))
        print("Personal Best Dist: " + str(personal_comp))

        global_best_dist = calculate_distance(self.current_position, global_best)
        random_vel = [round(uniform(0, 1), 4) for _ in range(vec_size)]
        global_vel = multiply_vectors(global_best_dist, random_vel)
        global_comp = scalar(global_vel, self.c2)
        print("Global Best: " + str(global_best))
        print("Global Best Dist: " + str(global_comp))

        velocity = calculate_velocity([inertia, personal_comp, global_comp])

        return velocity
    
class PSO:
    def __init__(self, swarm_size = 10, iterations=10, Wmin=0.4, Wmax=0.9, global_best = None, 
                 limit = limitations, printable=True):
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.swarm = []
        self.global_best = global_best
        self.Wmin = Wmin
        self.Wmax = Wmax
        self.limit = limit
        self.printable = printable
        self.best_performance = 0
    

    def start_csv(self, path, target):
        # preprocess
        global X, y, x_train, x_test, y_train, y_test
        df = pd.read_csv(path)
        X = df.drop(columns=target)
        y = df[target]
        number_of_classes = len(y.unique())

        preprocess = StructuredDataPreprocessing(X, y)
        X = preprocess.featuresEncoder()
        y = preprocess.labelncoder()

        init_hiperparams(number_of_classes)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        self.initialize_pso()

    def position_zero(self):
        position_zero = [0] * (self.limit[0])

        position_zero[0] = int((max(self.limit[1]) + min(self.limit[1])) / 2)
        position_zero[1] = int((max(self.limit[2]) + min(self.limit[2])) / 2)

        number_of_layers = int((max(self.limit[3]) + min(self.limit[3])) / 2)
        for layer in range(2, (3*number_of_layers), 3):
            position_zero[layer] = int((max(self.limit[4]) + min(self.limit[4])) / 2)
            position_zero[layer+1] = int((max(self.limit[5]) + min(self.limit[5])) / 2)
            position_zero[layer+2] = round(((max(self.limit[6]) + min(self.limit[6])) / 2), 4)
        
        return position_zero

    def generate_particle(self):
        current_position = [0] * (self.limit[0])

        current_position[0] = choice(self.limit[1])
        current_position[1] = choice(self.limit[2])

        number_of_layers = choice(self.limit[3])
        for layer in range(2, (3*number_of_layers), 3):
            current_position[layer] = (choice(self.limit[4]))
            current_position[layer+1] = (choice(self.limit[5]))
            current_position[layer+2] = round(uniform(self.limit[6][0], self.limit[6][1]), 4)

        current_vel = calculate_distance(self.position_zero(), current_position)
        personal_best = current_position

        c1 = round(uniform(0, 1), 4)
        c2 = round(uniform(0, 1), 4)

        return Particle(current_vel, current_position, personal_best, c1, c2)

    def generate_swarm(self):
        with open('PSO/results/pso_heart.txt', "w") as out:
            out.write("============================== Iteration 1 ==============================\n")

        for particle in range(self.swarm_size):
            p = self.generate_particle()
            self.swarm.append(p)

            performance = self.rate_performance(p.current_position)
            p.best_performance = performance
            if self.global_best == None or performance > self.best_performance:
                self.global_best = p.current_position
                self.best_performance = performance

            with open('PSO/results/pso_heart.txt', "a") as out:
                out.write("Particle " + str(particle) + " => 'Position':" + str(p.current_position) + " 'Vel': " + str(p.current_vel) + "\n" + 
                      "'Accuracy': " + str(performance) + "\n")

            print(p.current_position)
            print(performance)

        return self.swarm

    def optimizer_map(self, n):
        if n == 0:
            return "adam"
        elif n == 1:
            return "sgd"
    
    def activation_map(self, n):
        if n == 0:
            return "relu"
        elif n == 1:
            return "elu"

    def mlp_generator(self, architecture):
        model = Sequential()

        model.add(Dense(architecture[2], activation=self.activation_map(architecture[3]), input_shape=(X.shape[1],)))
        model.add(Dropout(architecture[4]))

        for layer in range(5, len(architecture), 3):
            if architecture[layer] != 0:
                model.add(Dense(architecture[layer], activation=self.activation_map(architecture[layer+1])))
                model.add(Dropout(architecture[layer+2]))

        model.add(Dense(output_size, activation=output_activation))

        model.compile(loss=loss_func, optimizer=self.optimizer_map(architecture[0]), metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=architecture[1], validation_data=(x_test, y_test))

        #file_path = "AutoDeepLearning\mlp_models\MLP_model_particle" + str(gen) + "_ind" + str(ind) + ".h5"
        #model.save(file_path)
        return model

    def rate_performance(self, architecture):
        model = self.mlp_generator(architecture)
        metrics = model.evaluate(x_test, y_test)
        return metrics[1]

    def move(self, particle, w):
        new_velocity = particle.next_velocity(w, self.global_best, len(particle.current_position))
        new_position = sum_vectors([new_velocity, particle.current_position])

        particle.current_vel = new_velocity
        particle.current_position = new_position

        performance = self.rate_performance(particle.current_position)

        particle.current_vel, particle.current_position, performance = self.reset_underperformation(performance, particle)

        if performance > particle.best_performance:
            if self.printable: print("New Personal Best!")
            particle.personal_best = particle.current_position
            particle.best_performance = performance

        if performance > self.best_performance:
            if self.printable: print("New Global Best!")
            self.global_best = particle.current_position
            self.best_performance = performance

        print(particle.current_position)
        print(performance)

        return [new_velocity, particle.current_position, performance]


    def reset_underperformation(self, performance, particle):
        if performance <= particle.best_performance or performance <= 0.5:
            new_position = self.generate_particle().current_position
            new_velocity = calculate_distance(particle.current_position, new_position)
            new_performance = self.rate_performance(new_position)

            return new_velocity, new_position, new_performance
        
        return particle.current_vel, particle.current_position, performance

    def initialize_pso(self):
        self.swarm = self.generate_swarm()

        with open('PSO/results/pso_heart.txt', "a") as out:
            out.write("\n")

        i = 2
        for w in np.linspace(self.Wmax, self.Wmin, self.iterations-1):
            with open('PSO/results/pso_heart.txt', "a") as out:
                out.write("============================== Iteration " + str(i) + " ==============================\n")
            j = 1
            for particle in self.swarm:
                vel_pos_per = self.move(particle, w)
                
                with open('PSO/results/pso_heart.txt', "a") as out:
                    out.write("Particle " + str(j) + " => 'Position':" + str(particle.current_position) + " 'Vel': " + str(particle.current_vel) + "\n" + 
                          "'Accuracy': " + str(vel_pos_per[2]) + "\n")
                j += 1
            
            i += 1

        with open('PSO/results/pso_heart.txt', "a") as out:
            out.write("\nThe best accuracy was: " + str(self.best_performance))

pso = PSO()

pso.start_csv(path="heart.csv", target="HeartDisease")