"""Function optimization using evolution strategies.

Minimize real-value single-objective functions using evolution strategy.

"""

import numpy as np
import matplotlib.pyplot as plt
from ec.es import OnePlusLambdaES, OnePlusOneES
from ec.benchmark import *


def minimize(function, runs):
    """Minimize function using (1+lambda)ES"""
    elite_fitness_runs = []
    for run in range(runs):
        print "Run: ", run
        
        my_es = OnePlusLambdaES()
        my_es.evolve(max_gen=200, 
                     genome_size=function.dim_, 
                     pop_size=100, 
                     fitness_func=function, 
                     domain_upper_bound=function.upper_bound_,
                     domain_lower_bound=function.lower_bound_,
                     verbose=True)
       
        elite_fitness_runs.append(np.array(my_es.staged_best_fitness_))
    
    all_mins = []
    all_maxs = []
    for run in range(runs):       
        plt.plot(elite_fitness_runs[run], marker='', label='Fitness')
        all_mins.append(elite_fitness_runs[run].min())
        all_maxs.append(elite_fitness_runs[run].max())      
        
    a = min(all_mins)
    b = max(all_maxs)
    plt.ylim([a-(b-a)*0.05, b+(b-a)*0.05])
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('(1+lambda)ES')
    plt.show()



###################################################################################################
if __name__ == '__main__':
    minimize(function=SphereFunction(), runs=10)
    #minimize(function=EasomFunction(), runs=10)

    
  

