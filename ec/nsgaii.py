"""Non-dominated Sorting Genetic Algorithm-II ( NSGA-II ) 

Deb, K.; Pratap, A.; Agarwal, S.; Meyarivan, T. (2002). "A fast and elitist multiobjective genetic 
algorithm: NSGA-II". IEEE Transactions on Evolutionary Computation 6 (2): 182.
"""

import numpy as np
from base import BaseEA
import copy


__all__=[
    "NsgaII"     
]


EPSILON = 1e-20 # For preventing such as dividing-by-zero 


class NsgaII(BaseEA):
    """Non-dominated Sorting Genetic Algorithm-II ( NSGA-II ) is a milestone-like evolutionary 
    algorithm of solving multi-objective optimization problems.
    
    Attributes:
    ----------
    pareto_front_ : Optimal solutions on the pareto front. 
    
    Reference:
    ----------
    Deb, K.; Pratap, A.; Agarwal, S.; Meyarivan, T. (2002). "A fast and elitist multiobjective 
    genetic algorithm: NSGA-II". IEEE Transactions on Evolutionary Computation 6 (2): 182.    
    """
    def __init__(self):
        super(NsgaII, self).__init__()
        
        # Attributes        
        self.pareto_front_ = None 
        self.pareto_solutions_ = None
        self.staged_pareto_front_ = None
        
        # Internal variables
        self.objs = None  # [n_indivs, n_objs]        
        self.fronts = None 
        
        self.pop_rank = None 
        self.pop_crowding_distance = None
        
        
    def _dominate(self, obj0, obj1):
        """Check dominance. 
        
        Parameters :
        -----------
        obj0 : a 1D array. [f0,f1,...,fn] smaller, better 
        
        obj1 : a 1D array. [f0,f1,...,fn] smaller, better
        
        Return :  1, if obj0 < obj1, obj0 dominates obj1 
                 -1, if obj0 > obj1, obj1 dominates obj0
                  0, otherwise
        """ 
        if not (False in (obj0 == obj1)):
            return  0
        elif not (False in (obj0 <= obj1)): 
            return  1
        elif not (False in (obj0 >= obj1)): 
            return -1
        else:
            return 0
          
        
    def _non_dominated_sort(self, objs):
        """Non-dominated sort. objs [n_points, n_objectives]            
        """
        n_points = objs.shape[0]
        rank = np.array([-1]*n_points)
        fronts = []
        worse_sets = [] 
        better_count = []
        front0 = []        
        for i in range(n_points):
            worse_than_i = []           # points that are dominated by i
            better_than_i_count = 0     # number of points that dominates i
            for k in range(n_points):
                flag = self._dominate(objs[i,:], objs[k,:])
                if flag == 1: worse_than_i.append(k)
                if flag == -1: better_than_i_count += 1    
                    
            worse_sets.append(worse_than_i)
            better_count.append(better_than_i_count)
        
            if better_than_i_count == 0:  # p(i) in front 0
                rank[i] = 1
                front0.append(i)
                
        fronts.append(front0)
        
        counter = 0        
        while fronts[-1]:
            next_front = []
            for k in fronts[-1]:
                for s in worse_sets[k]:
                    better_count[s] -= 1
                    if better_count[s] == 0:
                        rank[s] = counter + 1
                        next_front.append(s)
        
            counter += 1
            fronts.append(next_front)
            
        fronts.pop()
        return fronts, rank
    
    
    def _assign_crowding_distance(self, objs):
        """Calculate crowding distance"""
        n_points = objs.shape[0]
        n_objs = objs.shape[1]
        cds = np.array([0.0]*n_points)  # crowding distance 
        for i in range(n_objs):
            sorted_index = np.argsort(objs[:,i]) # ascending order
            cds[sorted_index[0]] = float('inf')
            cds[sorted_index[-1]] = float('inf')
            obj_i_min = objs[:,i].min()
            obj_i_max = objs[:,i].max()
            for k in range(1, n_points-1):                
                cds[sorted_index[k]] += (objs[sorted_index[k+1],i] - objs[sorted_index[k-1],i])  \
                                        /(obj_i_max - obj_i_min + EPSILON)
        return cds    
       
    
    def _breed(self):        
        """Overridden. Differential Evolution breed operator"""        
        super(NsgaII, self)._breed()                   
        # Differential Evolution breed operator
        self.offspring = np.empty_like(self.population)
        mutation_factor = 0.7
        crossover_prob = 0.25
        for i in range(self.population_size):
            xr1, xr2, xr3 = np.random.choice(self.population_size, 3, replace=False) 
            trial = self.population[xr1] + \
                        mutation_factor*(self.population[xr2] - self.population[xr3])            
            mask = (np.random.rand(1, self.genome_size) < crossover_prob).astype(float)
            self.offspring[i] = trial*mask + self.population[i]*(1-mask)            
                    
            
    def _tournament_select(self, tournament_size=2):        
        i, j = np.random.choice(self.population_size, 2, replace=False)
        return self._crowded_compare(i, j)         

    
    def _crowded_compare(self, i, j):
        if self.pop_rank[i] < self.pop_rank[j]:
            return i 
        elif self.pop_rank[i] > self.pop_rank[j]:
            return j
        elif self.pop_crowding_distance[i] > self.pop_crowding_distance[i]:
            return i
        else:
            return j
        

    def _select(self):
        """Overridden."""
        # Note that now FITNESS is a vector. 
        self.pop_fitness = self._evaluate(self.population)
        self.offspring_fitness = self._evaluate(self.offspring)        
        mating_pool = np.concatenate((self.population, self.offspring), axis=0)  
        mating_pool_objs = np.array(self.pop_fitness + self.offspring_fitness)        
                
        fronts, rank = self._non_dominated_sort(mating_pool_objs)
        self.fronts = fronts
        new_pop_index = []
        count = 0
        i = 0
        while count + len(fronts[i]) < self.population_size:
            count += len(fronts[i])
            new_pop_index += fronts[i]
            i += 1
               
        front_i = np.array(fronts[i])                  
        
        cds = self._assign_crowding_distance(mating_pool_objs[front_i,:]) 
        sorted_index = np.argsort(-cds)
        new_pop_index += front_i[sorted_index[0:self.population_size-count]].tolist()   

        self.pareto_front_ = np.array(mating_pool_objs)[np.array(self.fronts[0])]        
        self.population = mating_pool[new_pop_index]
        self.pop_rank = rank[new_pop_index]
        
        if len(self.fronts[0]) >= len(new_pop_index):
            self.pareto_solutions_ = copy.deepcopy(self.population)
        else:
            self.pareto_solutions_ = copy.deepcopy(mating_pool[self.fronts[0]])
        
        return self.population, self.pop_rank, self.pop_crowding_distance
                

    def _evaluate(self, population):
        """Overridden."""        
        pop_objs = []  # [n_indivs, n_objs]
        size = population.shape[0]
        for i in range(size):
            objs = self.fitness_func(population[i,:])
            pop_objs.append(objs)
        return pop_objs
      
       
    def _save_elite(self):
        """Save elite."""
        self.best_solution_ = self.pareto_solutions_
        self.best_fitness_ = self.pareto_front_
        self.staged_best_fitness_.append(copy.deepcopy(self.pareto_front_))



















