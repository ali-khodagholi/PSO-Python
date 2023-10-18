import time
import math
import pylab as py
from cost_fucntion import *

py.rcParams.update({'font.size': 14})

######################## Control parameters ########################
w = 0.5                   # Intertial weight. In some variations, it is set to vary with iteration number.
c1 = 2.0                  # Weight of searching based on the optima found by a particle
c2 = 2.0                  # Weight of searching based on the optima found by the swarm
v_fct = 1                 # Velocity adjust factor. Set to 1 for standard PSO.

nPop = 40                       # population size (number of particles)
nVar = 4                        # dimension (= no. of parameters in the fitness function)
max_iter = 100                  # maximum number of iterations 
var_max = np.zeros(nVar) - 4    # lower bound (does not need to be homogeneous)  
var_min = np.zeros(nVar) + 4    # upper bound (does not need to be homogeneous)   

######################## Defining and intializing variables ########################
pbest_val = np.zeros(nPop)            # Personal best fintess value. One pbest value per particle.
gbest_val = np.zeros(max_iter)      # Global best fintess value. One gbest value per iteration (stored).

pbest = np.zeros((nVar,nPop))            # pbest solution
gbest = np.zeros(nVar)                 # gbest solution

gbest_store = np.zeros((nVar,max_iter))   # storing gbest solution at each iteration

pbest_val_avg_store = np.zeros(max_iter)
fitness_avg_store = np.zeros(max_iter)

x = np.zeros((nVar,nPop))           # Initial position of the particles
v = np.zeros((nVar,nPop))                # Initial velocity of the particles

######################## Initial Position ########################
for m in range(nVar):    
    x[m,:] = var_max[m] + (var_min[m]-var_max[m]) * np.random.rand(nPop) 
    
######################## Main Loop ########################
t_start = time.time()
for iter in range(0,max_iter):
    
    if iter > 0:                             # nVaro not update postion for 0th iteration
        r1 = np.random.rand(nVar,nPop)            # random numbers [0,1], matrix nVar x nPop
        r2 = np.random.rand(nVar,nPop)            # random numbers [0,1], matrix nVar x nPop   
        v_global = np.multiply(((x.transpose()-gbest).transpose()),r2)*c2*(-1.0)    # velocity towards global optima
        v_local = np.multiply((pbest- x),r1)*c1           # velocity towards local optima (pbest)
    
        v = w*v + (v_local + v_global)       # velocity update
        x = x + v*v_fct                      # position update
    
    
    fit = fitness(x)                         # fitness function call (once per iteration). Vector nPop
    
    if iter == 0:
        pbest_val = np.copy(fit)             # initial personal best = initial fitness values. Vector of size nPop
        pbest = np.copy(x)                   # initial pbest solution = initial position. Matrix of size nVar x nPop
    else:
        # pbest and pbest_val update
        ind = np.argwhere(fit > pbest_val)   # indices where current fitness value set is greater than pbset
        pbest_val[ind] = np.copy(fit[ind])   # update pbset_val at those particle indices where fit > pbest_val
        pbest[:,ind] = np.copy(x[:,ind])     # update pbest for those particle indices where fit > pbest_val
      
    # gbest and gbest_val update
    ind2 = np.argmax(pbest_val)                       # index where the fitness is maximum
    gbest_val[iter] = np.copy(pbest_val[ind2])        # store gbest value at each iteration
    gbest = np.copy(pbest[:,ind2])                    # global best solution, gbest
    
    gbest_store[:,iter] = np.copy(gbest)              # store gbest solution
    
    pbest_val_avg_store[iter] = np.mean(pbest_val)
    fitness_avg_store[iter] = np.mean(fit)
    print("Iter. =", iter, ". gbest_val = ", gbest_val[iter])  # print iteration no. and best solution at each iteration
    

t_elapsed = time.time() - t_start
print("\nElapsed time = %.4f s" % t_elapsed)


######################## Plotting ########################
py.close('all')
py.plot(gbest_val,label = 'gbest_val')
py.plot(pbest_val_avg_store, label = 'Avg. pbest')
py.plot(fitness_avg_store, label = 'Avg. fitness')
py.legend()
#py.gca().set(xlabel='iterations', ylabel='fitness, gbest_val')

py.xlabel('iterations')
py.ylabel('fitness, gbest_val')


py.figure()
for m in range(nVar):
    py.plot(gbest_store[m,:],label = 'nVar = ' + str(m+1))
    
py.legend()
py.xlabel('iterations')
py.ylabel('Best solution, gbest[:,iter]')


        