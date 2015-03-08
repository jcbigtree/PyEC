"""Base classes for evolutionary computation.

Base class which implements the basic flow of evolutionary algorithms.

Warning. Should not be used directly. Use derived classes instead.
"""

import numpy as np
import matplotlib.pyplot as plt
import numbers
import copy
import unittest


class BaseEA(object):
    """Base class of all evolutionary algorithms. 
    
        ------------------------------------------------
        | The basic flow of Evolutionary Algorithm|
        ------------------------------------------------
        (1) INITIALIZE the population
        (2) While not-happy
           (2.1) EVALULATE the current population
           (2.2) SELECT a few individuals for generating offsprings
           (2.3) BREED
        (3) Output the best individual
        
    Attributes:
    -----------
    best_solution_: array-like. 
        Best solution found by an algorithm.
        
    best_fitness_: float. 
        Fitness of the best solution. 
        
    staged_best_solution_: a list of array.  
        Best solution at each generation.
        
    staged_best_fitness_: array.  
        Best fitness at each generation.    
        
    Warning: This class should not be used directly. Use derived class instead.
    """
    def __init__(self):
        # Attributes 
        self.best_solution_ = None
        self.best_fitness_ = None 
        self.staged_best_solution_ = []
        self.staged_best_fitness_ = []  
        
        # Parameters
        self.max_generation = 0
        self.genome_size = 0
        self.population_size = 0
        self.fitness_func = None
        self.domain_upper_bound = None
        self.domain_lower_bound = None

        # Internal variables
        self.population = None
        self.pop_fitness = None

        self.offspring = None
        self.offspring_fitness = None
        self.generation = 0
        self.elite_index = None

        #self.elite = None        
        #self.elite_fitness = None        
        #self.staged_elite = []
        #self.staged_elite_fitness = []
    

    def _initialize(self):
        """Randomly initialize population."""
        #np.random.seed()
        self.population = np.random.rand(self.population_size, self.genome_size)
        a = self.domain_lower_bound
        b = self.domain_upper_bound
        self.population = a + (b - a)*self.population


    def _breed(self):
        """Generate a new generation. Note that only very basic behaviors have been
            implemented here. This method must be called by the derived classes."""
        self.generation += 1
       
       
    def _regulate(self, population):
        """Regulate individuals if they go beyond the domain bounds.
        """
        for d in range(len(self.domain_lower_bound)): # Each dimension
            pop_d = population[:,d]
            pop_d[np.where(pop_d < self.domain_lower_bound[d])] = self.domain_lower_bound[d]
            pop_d[np.where(pop_d > self.domain_upper_bound[d])] = self.domain_upper_bound[d]
            population[:,d] = pop_d            
        return population
    
    
    def _evaluate(self, population):
        """Evaluates all the individuals, assigns fitness to each individual."""
        pop_fitness = []
        size = population.shape[0]
        for i in range(size):
            fitness = self.fitness_func(population[i,:])
            pop_fitness.append(fitness)
        return pop_fitness


    def _select(self):
        """Do nothing. Shall be overridden by derived classes."""
        pass


    def _save_elite(self):
        """Save elite."""
        pass 
    

    def _check_stop_criteria(self):
        """Check whether stop criteria is satisfied."""        
        return self.generation >= self.max_generation 
        

    def evolve(self,
               max_gen=None, 
               genome_size=None, 
               pop_size=None, 
               fitness_func=None, 
               domain_upper_bound=None,
               domain_lower_bound=None,
               verbose=False):
        """Evolve. Main loop of an evolutionary algorithm.
        
        Parameters:
        -----------
        max_gen : int
            maximum generation

        genome_size : int 
            Genome size
            
        pop_size : int 
            population size
          
        fitness_func. Object
            Fitness function.
            
        domain_upper_bound. numpy.ndarray of the shape [1, dim].
            Upper bound of the variables.The dimension should be the same as the problem dimenstion.
            
        domain_lower_bound. numpy.ndarray of the shape [1, dim].
            Lower bound of the variables.The dimension should be the same as the problem dimenstion.
            
        verbose. bool. Default False.
            Enable verbose output.
        """
        #Check parameters
        if (not isinstance(max_gen, (numbers.Integral, np.integer)) or max_gen < 1):
            raise ValueError(
                "max_gen must be a positive integer greater than 0"
                )

        if (not isinstance(genome_size, (numbers.Integral, np.integer)) or genome_size < 1):
            raise ValueError(
                "genome_size must be a positive integer greater than 0"
                )

        if (not isinstance(pop_size, (numbers.Integral, np.integer)) or pop_size < 1):
            raise ValueError(
                "pop_size must be a positive integer greater than 0"
                )

        if not hasattr(fitness_func, '__call__'):
            raise ValueError(
                "fitness_func must be a callable function"
                )                   
        
        if not (isinstance(domain_upper_bound, (np.ndarray, list, tuple)) and 
                isinstance(domain_upper_bound[0], numbers.Number)):
            raise ValueError(
                "domain_upper_bound must be a number array or list or tuple."
                )
        
        if not (isinstance(domain_lower_bound, (np.ndarray, list, tuple)) and 
                isinstance(domain_lower_bound[0], numbers.Number)):
            raise ValueError(
                "domain_lower_bound must be of type (numpy.ndarray, list or tuple)"
                )
            
        if not isinstance(verbose, bool):
            raise ValueError(
                "verbose must be True or False"
                )            
       
        self.max_generation = max_gen
        self.genome_size = genome_size
        self.population_size = pop_size
        self.fitness_func = fitness_func
        self.domain_upper_bound = np.array(domain_upper_bound)
        self.domain_lower_bound = np.array(domain_lower_bound)
        
        self.staged_best_fitness_ = []
        # Initialize population
        self._initialize()
        self.pop_fitness = self._evaluate(self.population)
        self._save_elite()

        while(not self._check_stop_criteria()): 
         
            # Generate offspring
            self._breed()     
            
            # Regulate offsprings
            self._regulate(self.offspring)       
            
            # Evaluate offspring
            self.offspring_fitness = self._evaluate(self.offspring)            
            
            # Select some individuals and form the new population
            self._select()
            
            # Save elite
            self._save_elite()
            
            # Verbose output           
            if verbose:
                print 'Iteration: ', self.generation
                print '  Best fitness: ', self.best_fitness_
                print '  Best solution: ', self.best_solution_

        return self.best_solution_, self.best_fitness_
    
