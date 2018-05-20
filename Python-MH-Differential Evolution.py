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
import pandas as pd
import numpy  as np
import random
import os

# Function: Initialize Variables
def initial_position(n = 3, min_values = [-5,-5], max_values = [5,5]):
    position = pd.DataFrame(np.zeros((n, len(min_values))))
    position['Fitness'] = 0.0
    for i in range(0, n):
        for j in range(0, len(min_values)):
             position.iloc[i,j] = random.uniform(min_values[j], max_values[j])
        position.iloc[i,-1] = target_function(position.iloc[i,0:position.shape[1]-1])
    return position

# Function: Velocity
def velocity(position, best_global, k0 = 0, k1 = 1, k2 = 2, F = 0.9, min_values = [-5,-5], max_values = [5,5], Cr = 0.2):
    v = best_global.copy(deep = True)
    for i in range(0, len(best_global)):
        ri = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        if (ri <= Cr):
            v[i] = best_global[i] + F*(position.iloc[k1, i] - position.iloc[k2, i])
        else:
            v[i] = position.iloc[k0, i]
        if (i < len(min_values) and v[i] > max_values[i]):
            v[i] = max_values[i]
        elif(i < len(min_values) and v[i] < min_values[i]):
            v[i] = min_values[i]
    v[-1] = target_function(v[0:len(min_values)])
    return v

# DE Function. DE/Best/1/Bin Scheme.
def differential_evolution(n = 3, min_values = [-5,-5], max_values = [5,5], iterations = 50, F = 0.9, Cr = 0.2):    
    count = 0
    position = initial_position(n = n, min_values = min_values, max_values = max_values)
    best_global = position.iloc[position['Fitness'].idxmin(),:].copy(deep = True)

    while (count <= iterations):
        print("Iteration = ", count, " of ", iterations)
        for i in range(0, position.shape[0]):
            k1 = int(np.random.randint(position.shape[0], size = 1))
            k2 = int(np.random.randint(position.shape[0], size = 1))
            while k1 == k2:
                k1 = int(np.random.randint(position.shape[0], size = 1))
            vi = velocity(position, best_global, k0 = i, k1 = k1, k2 = k2, F = F, min_values = min_values, max_values = max_values, Cr = Cr)
           
            if (vi[-1] <= position.iloc[i,-1]):
                for j in range(0, position.shape[1]):
                    position.iloc[i,j] = vi[j]
            if (best_global[-1] > position.iloc[position['Fitness'].idxmin(),:][-1]):
                best_global = position.iloc[position['Fitness'].idxmin(),:].copy(deep = True)  
            

        count = count + 1 
        
    print(best_global)    
    return best_global

######################## Part 1 - Usage ####################################

# Function to be Minimized. Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def target_function (variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

de = differential_evolution(n = 15, min_values = [-5,-5], max_values = [5,5], iterations = 100, F = 0.9, Cr = 0.2)
