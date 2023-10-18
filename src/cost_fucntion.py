import numpy as np

# Fitness function. The code maximizes the value of the fitness function
def fitness(x):
    F_sphere = 2.0 - np.sum(np.multiply(x,x),0)    # modified sphere function
    return F_sphere