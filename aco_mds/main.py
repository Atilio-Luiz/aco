import networkx as nx
from aco_mds import aco_mds

# parameters of the algorithm
number_of_iterations = 100
number_of_ants = 7
evaporation_rate = 0.2
d_aco_rate = 0.7
d_rate = 0.9
rate_augmentation = 0.3
max_noimpr = 10
k_max = 5
destroy_min = 0.2
destroy_max = 0.5
rvns_max_itr = 37

def calculate_dominating_set(graph, name_of_the_graph):
    print("--------------------------")
    print(name_of_the_graph,"with", len(graph), "vertices:")
    obj = aco_mds(graph,number_of_iterations,number_of_ants,evaporation_rate,d_aco_rate,
                d_rate,rate_augmentation,max_noimpr,k_max,destroy_min,destroy_max,rvns_max_itr)
    dominating_set = obj.run()
    print("Dominating set with",len(dominating_set),"vertices:", dominating_set)

# calculate the (approximate) minimal domination set of some cycles
for n in range(3,10):
    G = nx.cycle_graph(n)
    calculate_dominating_set(G, "Cycle")

# calculate the (approximate) minimal domination set of some stars
for n in range(2,9):
    G = nx.star_graph(n)
    calculate_dominating_set(G, "Star")

# calculate the (approximate) minimal domination set of some paths
for n in range(2,10):
    G = nx.path_graph(n)
    calculate_dominating_set(G, "Path")

# calculate the (approximate) minimal domination set of some complete graphs
for n in range(2,10):
    G = nx.complete_graph(n)
    calculate_dominating_set(G, "Complete")

# calculate the (approximate) minimal domination set of some wheel graphs
for n in range(4,20):
    G = nx.wheel_graph(n)
    calculate_dominating_set(G, "Wheel")