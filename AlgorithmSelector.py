import GraphAndFileProcess
from GraphAndFileProcess import Graph
from AlgorithmsImplementation import GraphProcessor
from AlgorithmsImplementation import k_clique
from AlgorithmsImplementation import max_clique
from AlgorithmsImplementation import DBK_Algorithm
from AlgorithmsImplementation import bro_kerbosch_algorithm
class AlgorithmSelector:
    def __init__(self):
        # 初始化算法字典
        self.algorithms = {
            "max_clq": max_clique,
            "Bron-Kerbosch": bro_kerbosch_algorithm,
            "k-clique": k_clique,
            #"clisat": self.clisat,
            'dbk': DBK_Algorithm
        }

    def register_algorithm(self, algorithm_name: str, algorithm_class):
        """注册算法，通过名称和类来注册"""
        self.algorithms[algorithm_name] = algorithm_class

    def select_algorithm(self, algorithm_name: str, graph, **params):
        """选择并返回相应的算法实例，并传递参数"""
        if algorithm_name in self.algorithms:
            algorithm_class = self.algorithms[algorithm_name]
            algorithm_instance = algorithm_class(graph)
            return algorithm_instance.run(**params)  # 使用 run 方法并传递额外参数
        else:
            raise ValueError(f"算法 {algorithm_name} 未找到.")

