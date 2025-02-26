from AlgorithmSelector import AlgorithmSelector
from CLQReader import CLQReader
from GraphAndFileProcess import Graph
from AlgorithmsImplementation import GraphProcessor

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
    print("fighting!")
