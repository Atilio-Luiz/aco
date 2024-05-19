########################################################################################
# Algorithm ACO-MDS
#
# This algorithm aims to calculate a minimum dominating set 
# of a given simple undirected graph G.
# However, an optimal solution is not guaranteed.
#
# This algorithm is based on Ant Colony Optimization (ACO) meta-heuristic,
# implemented using the MAX-MIN Ant System (MMAS) and the Hyper-Cube Framework (HCF).
# This algorithm also uses, as a local search step, the heuristic called
# Reduced Variable Neighborhood Search (RVNS).
#
# Author: Atílio Gomes Luiz
# Date  : May 19th, 2024
#
# This algorithm is inspired on other algorithms contained in the following two papers: 
# Paper 1: https://www.sciencedirect.com/science/article/abs/pii/S1568494619302200
# Paper 2: https://www.sciencedirect.com/science/article/abs/pii/S1568494624000802
# The above two papers are based on theory that was developed in the following papers:
# Paper 3: https://ieeexplore.ieee.org/abstract/document/1275547
# Paper 4: https://www.sciencedirect.com/science/article/abs/pii/S0167739X00000431
# Paper 5: https://www.sciencedirect.com/science/article/abs/pii/S0305054897000312
########################################################################################
import random

class aco_mds:
    ##
    # constructor
    ##
    def __init__(self, graph, number_of_iterations, number_of_ants,evaporation_rate,
                 d_aco_rate,d_rate,rate_augmentation,max_noimpr,k_max,destroy_min,
                 destroy_max,rvns_max_itr):
        """ Parameters of the aco_mds algorithm """
        # graph G
        self.G = graph
        # total number of iterations
        self.number_of_iterations = number_of_iterations
        # number of solution constructions per iteration in ACO
        self.number_of_ants = number_of_ants 
        # evaporation rate of ACO
        self.evaporation_rate = evaporation_rate
        # a constant for selecting vertices in the construct_solution
        self.d_aco_rate = d_aco_rate
        # a constant for choosing vertices in the extend_solution (also used in RVNS)
        self.d_rate = d_rate
        # constant parameter to iteratively add the number of vertices in each step
        # of the extend_solution function
        self.rate_augmentation = rate_augmentation
        # maximum number of consecutive iterations without improvement of current
        # solution used in Reduced Variable Neighborhood Search (RVNS)
        self.max_noimpr = max_noimpr
        # number of neighborhood functions in Reduced Variable Neighborhood Search (RVNS)
        self.k_max = k_max
        # minimum value of destruction rate applied within the 
        # neighborhood function context in Reduced Variable Neighborhood Search (RVNS)
        self.destroy_min = destroy_min
        # maximum value of destruction rate applied within the 
        # neighborhood function context in Reduced Variable Neighborhood Search (RVNS)
        self.destroy_max = destroy_max
        # maximum number of iterations for Reduced Variable Neighborhood Search (RVNS)
        self.rvns_max_itr = rvns_max_itr

        """ Additional atributes """
        # saves the best solution in the current iteration 
        self.iteration_best_solution = set()
        # saves the best solution so far
        self.best_so_far_solution = set()
        # convergence factor
        self.covergence_factor = 0
        # list of pheromones for each vertex
        self.pheromone = [-1] * len(self.G)



    ##
    # ant colony optimization main routine
    ##
    def run(self):
        self.covergence_factor = 0
        self.best_so_far_solution = set(range(0,len(self.G)))
        self.initialize_pheromone_values()
        counter = self.number_of_iterations
        while counter > 0:
            self.iteration_best_solution = set(range(0,len(self.G)))
            for i in range(0, self.number_of_ants):
                S = set()
                S = self.construct_feasible_solution(S, self.aco_choose_from)
                S = self.extend_solution(S)
                S = self.reduce_solution(S)
                S = self.reduced_variable_neighborhood_search(S)
                if len(S) < len(self.iteration_best_solution):
                    self.iteration_best_solution = S
            
            if self.iteration_best_solution < self.best_so_far_solution:
                self.best_so_far_solution = self.iteration_best_solution

            self.update_pheromone()
            self.compute_convergence_factor()

            if self.covergence_factor > 0.99:
                self.initialize_pheromone_values()
            
            counter = counter - 1

        return self.best_so_far_solution


    ##
    # function that returns the degree of a vertex v in the graph self.G
    ##
    def deg(self, v):
        return len(self.G[v])

    ##
    # function that initialize all pheromone values to 0.5
    ##
    def initialize_pheromone_values(self):
        for i in range(0, len(self.pheromone)):
            self.pheromone[i] = 0.5


    ##
    # function that receives a (possibly empty) 
    # partial dominating set S of graph self.G and returns
    # a valid dominating set of graph self.G
    ##
    def construct_feasible_solution(self, S, func_choose_vertex):
        remaining_vertices = set()
        if len(S) == 0:
            remaining_vertices = set(self.G)
        else:
            remaining_vertices = set(self.G).difference(self.closed_neighborhood_of_set(S))
        while len(self.closed_neighborhood_of_set(S)) < len(self.G):
            v = func_choose_vertex(remaining_vertices)
            S.add(v)
            remaining_vertices = set(self.G) - self.closed_neighborhood_of_set(S)
        return S

    ##
    # function that receives a (possibly empty) set of vertices S and 
    # returns a set containing the closed neighborhood of S
    ##
    def closed_neighborhood_of_set(self,S):
        neighbours = set(S)
        for v in S:
            for ngb in self.G[v]:
                neighbours.add(ngb)
        return neighbours
    
    ##
    # function that extends a given dominating set S of graph self.G
    # by appending additional vertices to S
    ##
    def extend_solution(self,S):
        # defines the number of vertices to be added to the set S
        number_of_vertices = int(min(self.rate_augmentation*len(S), len(set(self.G)-S)))
        # add new vertices using one of two different strategies, 
        # the strategy is chosen according to the value of variables r and self.d_rate, 
        # where the variable r is chosen probabilistically and self.d_rate is a 
        # parameter of the algorithm set in the constructor.
        for i in range(0, number_of_vertices):
            r = random.random()
            remaining_vertices = list(set(self.G) - S)
            random.shuffle(remaining_vertices)
            if r <= self.d_rate:
                # chooses a vertex with maximum degree
                maxDegree = 0
                chosenVertex = -1
                for v in remaining_vertices:
                    if self.deg(v) > maxDegree:
                        maxDegree = self.deg(v)
                        chosenVertex = v
                S.add(chosenVertex)
            else:
                # selection of vertex by roulette-wheel 
                # vertices with larger degree have greater probability of being chosen
                list_of_pairs = [[v, self.deg(v)] for v in remaining_vertices]
                chosenVertex = self.roullete_wheel_selection(list_of_pairs)
                S.add(chosenVertex)

        return S
    

    ##
    # function that receives a list of pairs: (vertex, probability)
    # and selects a vertex using the roullete wheel selection method
    ##
    def roullete_wheel_selection(self, pairs_vertex_probability):
        # sorts the pairs by probability
        pairs_vertex_probability.sort(key=lambda x: x[1])
        # Implementation of Roullete Wheel Selection
        # Step 1: calculate the total fitness
        total_fitness = 0
        for i in range(0, len(pairs_vertex_probability)):
            total_fitness = total_fitness + pairs_vertex_probability[i][1]
        # Step 2: Calculate the selection probability for each individual
        probabilities = [0] * len(pairs_vertex_probability)
        for i in range(0, len(pairs_vertex_probability)):
            probabilities[i] = pairs_vertex_probability[i][1] / total_fitness
        # Step 3: Compute the cumulative probability
        cumulative_probabilities = [0] * len(pairs_vertex_probability)
        cumulative_sum = 0
        for i in range(0, len(pairs_vertex_probability)):
            cumulative_sum += probabilities[i]
            cumulative_probabilities[i] = cumulative_sum
        # Step 4: Generate a random number between 0 and 1
        random_number = random.random()
        # Step 5: Select the individual based on the random number
        for i in range(0, len(pairs_vertex_probability)):
            if cumulative_probabilities[i] >= random_number:
                return pairs_vertex_probability[i][0]


    ##
    # function that receives a set of vertices S
    # and chooses one vertex from S by means of a probabilistic approach
    ##
    def aco_choose_from(self,S):
        # generate a random number between 0 and 1
        r = random.random()
        # calculates the value f(v) = deg(v)*pheromone[v] for each vertex v in S
        f = dict()
        for v in S:
            f[v] = self.deg(v) * self.pheromone[v]
        # select a vertex v from S that maximizes 
        # the objective function f(v) = deg(v) * pheromone[v]
        if r <= self.d_aco_rate:
            best_vertex = -1
            best_value = -1
            for v in S:
                if f[v] > best_value:
                    best_value = f[v]
                    best_vertex = v
            return best_vertex
        else:
            # selects the vertex by using roullete wheel method
            list_of_pairs = [[v, f[v]] for v in S]
            return self.roullete_wheel_selection(list_of_pairs)
            
            

    ##
    # function that receives a dominating set S of the graph self.G
    # and try to reduces the size of S maintaining the resulting
    # set as a dominating set of self.G
    # In this context, a vertex v in S is called 'redundant' if
    # all vertices from its closed neighborhood N[v] are dominated
    # by vertices from S. Therefore, each such redundant vertex v
    # may safely be removed from S. This is done iteratively, one after the other.
    ##
    def reduce_solution(self, S):
        final_solution = S.copy()
        for v in S:
            v_is_not_dominated = True
            for w in final_solution:
                if w != v and v in self.G[w]:
                    v_is_not_dominated = False
                    break
            if v_is_not_dominated:
                continue
            
            some_neighbor_not_dominated = False
            for ngh in self.G[v]:
                ngh_is_dominated = False
                if ngh in final_solution:
                    ngh_is_dominated = True
                for x in final_solution:
                    if x != v and ngh in self.G[x]:
                        ngh_is_dominated = True
                if not ngh_is_dominated:
                    some_neighbor_not_dominated = True
                    break
            if some_neighbor_not_dominated:
                continue
                
            final_solution.remove(v)
        
        return final_solution

    ##
    # function that updates the pheromone of each vertex of the graph self.G
    ##
    def update_pheromone(self):
        if self.covergence_factor < 0.4:
            k_ib = 1
            k_bsf = 0
        elif self.covergence_factor >= 0.4 and self.covergence_factor < 0.6:
            k_ib = 2/3
            k_bsf = 1/3
        elif self.covergence_factor >= 0.6 and self.covergence_factor < 0.8:
            k_ib = 1/3
            k_bsf = 2/3
        else:
            k_ib = 0
            k_bsf = 1
        for v in self.G:
            delta_ib = delta_bsf = 0
            if v in self.iteration_best_solution:
                delta_ib = 1
            if v in self.best_so_far_solution:
                delta_bsf = 1
            epsilon = k_ib*delta_ib + k_bsf*delta_bsf
            self.pheromone[v] = self.pheromone[v] + self.evaporation_rate*(epsilon-self.pheromone[v])
            if self.pheromone[v] > 0.999:
                self.pheromone[v] = 0.999
            if self.pheromone[v] < 0.001:
                self.pheromone[v] = 0.001
            
    
    ##
    # function that calculates the convergence factor.
    # covergence_factor is calculated based on the current pheromone values.
    # When all pheromone values are set to 0.5, we have that covergence_factor = 0.
    # When all pheromone values are either 0.001 or 0.999, then covergence_factor = 1.
    # In all other cases, covergence_factor has a value between 0 and 1.
    # In other words, the higher the value of covergence_factor, the closer is the
    # algorithm to convergence.
    ##
    def compute_convergence_factor(self):
        pmax = 0.999
        pmin = 0.001
        numerator = sum([max(pmax-self.pheromone[u],self.pheromone[u]-pmin) for u in self.G])
        denominator = len(self.pheromone) * (pmax - pmin)
        self.covergence_factor = 2 * (numerator/denominator) - 1
                
    ##############################################################################################
    ## Below are the functions of the meta-heuristic RVNS
    ##############################################################################################

    ##
    # A reduced variant of Variable Neighborhood Search (VNS) Metaheuristic.
    # This variant is obtained by removing the local search phase from VNS.
    ##
    def reduced_variable_neighborhood_search(self,S):
        k = 1
        c_noimpr = 0
        max_itr = self.rvns_max_itr

        while c_noimpr < self.max_noimpr and max_itr > 0:
            Sp = self.destroy_solution(S, k)
            Sp = self.construct_feasible_solution(Sp, self.rvns_choose_from)
            Sp = self.extend_solution(Sp)
            Sp = self.reduce_solution(Sp)
            if len(Sp) < len(S):
                S = Sp
                k = 1
                c_noimpr = 0
            else:
                k += 1
                c_noimpr += 1
                if k > self.k_max:
                    k = 1
            
            max_itr = max_itr - 1

        return S
    
    ##
    # function that receives a dominating set S of the graph self.G
    # and destroy it by removing some vertices from S.
    # The quantity of removed vertices is calculated according to the value 
    # of the parameters self.destroy_min, self.destroy_max, self.k_max, and self.d_rate
    ##
    def destroy_solution(self, S, k):
        n_iterations = self.destruction_rate(k) * len(S)

        while n_iterations > 0 and len(S) > 0:
            random_vertex = random.choice(list(S))
            S.remove(random_vertex)
            n_iterations = n_iterations - 1
        
        return S

    ##
    # destruction rate
    ##
    def destruction_rate(self, k):
        return self.destroy_min + (k-1)*((self.destroy_max-self.destroy_min)/(self.k_max-1))
    
    
    ##
    # function that receives a set of vertices S
    # and chooses one of the vertices of S by means of a probabilistic approach.
    # This function is used by the Reduced Variable Neighborhood Search (RVNS)
    ##
    def rvns_choose_from(self, S):
        # chooses a random number between 0 and 1
        r = random.random()
        # two ways of choosing a vertex, based on the value of random number r
        if r <= self.d_rate:
            # calculates the white degrees of the vertices in S
            list_of_pairs = list()
            for v in S:
                v_white_degree = self.deg(v)
                for ngh in self.G[v]:
                    for w in S:
                        if w != v and ngh in self.G[w]:
                            v_white_degree = v_white_degree - 1
                            break
                list_of_pairs.append([v, v_white_degree])
            # chooses a vertex with maximum white degree     
            max_degree = -1
            chosenVertex = -1
            for i in range(0,len(list_of_pairs)):
                if list_of_pairs[i][1] > max_degree:
                    max_degree = list_of_pairs[i][1]
                    chosenVertex = list_of_pairs[i][0]
            return chosenVertex
        else:
            # chooses a random number from S     
            return random.choice(list(S))