"""Demo NSGA-II
"""

import matplotlib.pyplot as plt
from ec.nsgaii import *
from ec.benchmark import ZDT2Function


def minimize(function, runs):
    """Minimize function using NSGA-II"""
    #elite_fitness_runs = []
    for run in range(runs):
        print "Run: ", run

        my_nsga2 = NsgaII()
        my_nsga2.evolve(max_gen=100, 
                        genome_size=function.dim_, 
                        pop_size=50, 
                        fitness_func=function, 
                        domain_upper_bound=function.upper_bound_,
                        domain_lower_bound=function.lower_bound_,
                        verbose=True)
        
        #elite_fitness_runs.append(np.array(my_es.staged_best_fitness_))
        pf = my_nsga2.pareto_front_
        plt.plot(pf[:,0], pf[:,1],'go')
        plt.show()

###################################################################################################
if __name__ == '__main__':
    minimize(function=ZDT2Function(), runs=1)

    
  

