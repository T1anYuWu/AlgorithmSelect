from AlgorithmSelector import AlgorithmSelector
from CLQReader import CLQReader
from GraphAndFileProcess import Graph
from AlgorithmsImplementation import GraphProcessor
import random
if __name__ == '__main__':
    # init algorithm selector
    algortrihm_seletor = AlgorithmSelector()
    
    # create Graph
    #graph = Graph()

    # get graph from DIMACS file
    #graph.parse_dimacs("dataset/dimacs/brock200_1.clq")
    """"测试用例
    graph = Graph()
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    #graph.add_edge(2, 3)
    graph.add_edge(2, 4)
    #graph.add_edge(3, 4)
    graph.add_edge(3, 5)
    graph.add_edge(3, 6)
    graph.add_edge(4, 5)
    #graph.add_edge(3,6)"""
    """用例2"""
    graph = Graph()
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)

    """用例3
    # 创建一个图对象
    graph = Graph()

    # 随机生成包含 20 个节点的图
    num_nodes = 10
    probability = 0.5 # 连接的概率，您可以调整这个值来控制图的密度

    # 随机添加边
    for i in range(1, num_nodes + 1):
        for j in range(i + 1, num_nodes + 1):
            if random.random() < probability:  # 以一定的概率连接节点
                graph.add_edge(i, j)

    # 输出生成的图的信息
    #print(graph)"""

    print(f"构建的图: {graph}")
    # select algorithm calculate max clique
    #max_clique = algortrihm_seletor.select_algorithm('max_clq',graph)
    #max_clique = algortrihm_seletor.select_algorithm('Bron-Kerbosch', graph)
    #max_clique = algortrihm_seletor.select_algorithm('k-clique', graph, k_max=None)
    #max_clique = algortrihm_seletor.select_algorithm('clisat', graph)
    max_clique = algortrihm_seletor.select_algorithm('dbk', graph)
    print("最大团的顶点集合为：", max_clique)
    print(f"最大团的大小为:{len(max_clique)}")

    """ 批量读文件测试代码"""
    """
    directory_path = 'dataset/evil2-main/evil-basic_graphs'  # 替换为实际文件夹路径
    clq_reader = CLQReader(directory_path)

    # 读取所有 CLQ 文件中的图
    clq_reader.parse_clq_files()

    # 获取读取到的图集合
    graphs = clq_reader.get_graphs()

    print(graphs)"""
