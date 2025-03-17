import os

from GraphAndFileProcess import Graph


class CLQReader:
    def __init__(self,directory_path):
        self.directory_path = directory_path
        self.graphs = {}

    def parse_clq_files(self):
        """ 从指定文件夹中读取所有CLQ文件"""
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".clq"):
                file_path = os.path.join(self.directory_path, filename)
                graph = Graph()
                graph.parse_dimacs(file_path)
                if graph.nodes:
                    self.graphs[filename] = graph

    def get_graphs(self):
        """返回以文件名为键，图对象为值的字典"""
        return self.graphs