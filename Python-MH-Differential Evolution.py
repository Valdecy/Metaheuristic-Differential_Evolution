############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Differential Evolution

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Differential_Evolution, File: Python-MH-Differential Evolution.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Differential_Evolution>

############################################################################

# Required Libraries
import math
import numpy  as np
import random
import os

# Function
def target_function():
    return

# Function: Initialize Variables
def initial_position(n = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((n, len(min_values) + 1))
    for i in range(0, n):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

# Function: Velocity
def velocity(position, best_global, k0 = 0, k1 = 1, k2 = 2, F = 0.9, min_values = [-5,-5], max_values = [5,5], Cr = 0.2, target_function = target_function):
    v = np.copy(best_global)
    for i in range(0, len(best_global)):
        ri = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        if (ri <= Cr):
            v[i] = best_global[i] + F*(position[k1, i] - position[k2, i])
        else:
            v[i] = position[k0, i]
        if (i < len(min_values) and v[i] > max_values[i]):
            v[i] = max_values[i]
        elif(i < len(min_values) and v[i] < min_values[i]):
            v[i] = min_values[i]
    v[-1] = target_function(v[0:len(min_values)])
    return v

# DE Function. DE/Best/1/Bin Scheme.
def differential_evolution(n = 3, min_values = [-5,-5], max_values = [5,5], iterations = 50, F = 0.9, Cr = 0.2, target_function = target_function):    
    count = 0
    position = initial_position(n = n, min_values = min_values, max_values = max_values, target_function = target_function)
    best_global = np.copy(position [position [:,-1].argsort()][0,:])
    while (count <= iterations):
        print("Iteration = ", count)
        for i in range(0, position.shape[0]):
            k1 = int(np.random.randint(position.shape[0], size = 1))
            k2 = int(np.random.randint(position.shape[0], size = 1))
            while k1 == k2:
                k1 = int(np.random.randint(position.shape[0], size = 1))
            vi = velocity(position, best_global, k0 = i, k1 = k1, k2 = k2, F = F, min_values = min_values, max_values = max_values, Cr = Cr, target_function = target_function)        
            if (vi[-1] <= position[i,-1]):
                for j in range(0, position.shape[1]):
                    position[i,j] = vi[j]
            if (best_global[-1] > position [position [:,-1].argsort()][0,:][-1]):
                best_global = np.copy(position [position [:,-1].argsort()][0,:])  
        count = count + 1 
    print(best_global)    
    return best_global

######################## Part 1 - Usage ####################################

# Function to be Minimized (Six Hump Camel Back). Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def six_hump_camel_back(variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

de = differential_evolution(n = 15, min_values = [-5,-5], max_values = [5,5], iterations = 100, F = 0.5, Cr = 0.2, target_function = six_hump_camel_back)

# Function to be Minimized (Rosenbrocks Valley). Solution ->  f(x) = 0; xi = 1
def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value

de = differential_evolution(n = 25, min_values = [-5,-5,-5], max_values = [5,5,5], iterations = 1000, F = 0.5, Cr = 0.2, target_function = rosenbrocks_valley)
