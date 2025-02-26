from GraphAndFileProcess import Graph
from AlgorithmsImplementation import GraphProcessor


class AlgorithmSelector:
    def __init__(self):
        # init algorithms
        self.algorithms = {
            #"max_clq": self.max_clq,
            #"max_cliqueDyn": self.max_cliquedyn,
            "Bron-Kerbosch": self.bron_kerbosch
            # Add algorithm here
        }

    def select_algorithm(self, algorithm_name: str, graph: Graph):
        # algorithms selector
        graph_processor = GraphProcessor(graph)
        if algorithm_name in self.algorithms:
            return self.algorithms[algorithm_name](graph_processor)
        else:
            raise ValueError(f"Algorithm {algorithm_name} not found.")

    def bron_kerbosch(self, graph_processor: GraphProcessor):
        """选择Bron-Kerbosch算法"""
        return graph_processor.bron_kerbosch()