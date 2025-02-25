from AlgorithmSelector import AlgorithmSelector
from Graph import Graph
from GraphProcessor import GraphProcessor

if __name__ == '__main__':
    # init algorithm selector
    algortrihm_seletor = AlgorithmSelector()
    
    # create Graph
    graph = Graph()

    # get graph from DIMACS file
    graph.parse_dimacs("dataset/DIMACS_subset_ascii/brock200_2.clq")

    print(f"构建的图: {graph}")
    # select algorithm calculate max clique
    max_clique = algortrihm_seletor.select_algorithm('Bron-Kerbosch',graph)
    print(max_clique)
