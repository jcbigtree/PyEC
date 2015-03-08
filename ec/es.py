"""(1 + lambda) Evolution Strategy

Evolution Strategies are a special kind of evolutionary algorithms which generally only 
mutation-like operators are employed. Here two special ES, (1+lambda)ES and (1+1)ES are 
implemented.

"""

import numpy as np
import copy
from base import BaseEA


__all__=[
    "OnePlusLambdaES", "OnePlusOneES"     
]

class OnePlusLambdaES(BaseEA):
    """(1 + lambda) Evolution Strategy"""
    def __init__(self):
        super(OnePlusLambdaES, self).__init__()
        self.step_size = 1.0
        self.step_succeeded = False
        self.staged_step_size = []        


    def _breed(self):
        """Overriden. Generate a whole population based on the best-so-far individual."""
        super(OnePlusLambdaES, self)._breed()   # call base method
        randn01_data = np.random.randn(self.population_size, self.genome_size)
        self.offspring = self.best_solution_ + self.step_size * randn01_data


    def _select(self):
        """Overriden."""
        # Find the best individual in the offspring
        k = np.argmax(self.offspring_fitness)
        # Check if it is better than the one in the current population
        if self.pop_fitness[0] < self.offspring_fitness[k]:
            self.population[0,:] = self.offspring[k,:]
            self.pop_fitness[0] = self.offspring_fitness[k]
            self.step_succeeded = True
        else:
            self.step_succeeded = False
                
        # Adjust step - size
        self._auto_adjust_stepsize(method=2)


    def _check_stop_criteria(self):
        """Overriden."""
        # If it exceeds the max iteration, stop the evolution.
        stop_flag_0 = super(OnePlusLambdaES, self)._check_stop_criteria()

        # If step_size is smaller than a threshold, stop the evolution.
        stop_flag_1 =  abs(self.step_size) < np.exp(-10)

        return stop_flag_0 or stop_flag_1


    def _auto_adjust_stepsize(self, method):
        """Adjust step size adaptively"""
        if method == 0: # One implementation of the 1/5 rule
            d = np.sqrt(self.genome_size + 1)
            if self.step_succeeded:
                self.step_size *= np.exp(1.0/d)
            else:
                self.step_size /= np.exp(1.0/d)

        if method == 1: # Generation-based
            if self.generation < self.max_generation:
                dmr = self.step_size/(self.max_generation - self.generation)
                self.step_size -= dmr
            else:
                self.step_size = 0

        if method == 2: # Another implementation of the 1/5 rule
            d = np.sqrt(self.genome_size + 1)
            if self.step_succeeded:
                u = 0.8
            else:
                u = -0.2
            self.step_size *= np.exp(1.0/d*u)

        self.staged_step_size.append(self.step_size)
        
        
    def _save_elite(self):
        """Save elite"""
        self.elite_index = np.argmax(self.pop_fitness)
        self.best_solution_ = self.population[self.elite_index,:]
        self.best_fitness_ = self.pop_fitness[self.elite_index]
        self.staged_best_solution_.append(self.best_solution_)
        self.staged_best_fitness_.append(self.best_fitness_)


####################################################################################################
class OnePlusOneES(OnePlusLambdaES):
    """(1 + 1) Evolution Strategy"""
    def __init__(self):
        """Initialization. 
        
        Note that the population size is fixed to 1.
        """
        super(OnePlusOneES, self).__init__()
        self.population_size = 1

        



