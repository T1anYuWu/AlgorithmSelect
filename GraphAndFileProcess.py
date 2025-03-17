import networkx as nx
import numpy as np


class Graph:
    def __init__(self):
        self.nodes = set()  # node Set
        self.edges = {}  # edge Set

    def add_node(self, node):
        # add_node
        if node not in self.nodes:
            self.nodes.add(node)
            self.edges[node] = set()

    def add_edge(self, node1, node2):
        # add one edge for node1 and node2
        self.add_node(node1)
        self.add_node(node2)
        self.edges[node1].add(node2)
        self.edges[node2].add(node1)

    def get_neighbors(self, node):
        # 获取节点的邻居
        return self.edges.get(node, set())

    def remove_node(self, node):
        """移除节点及其相关的所有边"""
        if node in self.nodes:
            # 移除所有相邻节点的连接
            for neighbor in list(self.edges[node]):
                self.edges[neighbor].remove(node)

            # 移除节点本身
            self.nodes.remove(node)
            del self.edges[node]

    def select_min_degree_vertex(self):
        """选择度数最小的顶点"""
        return min(self.nodes, key=lambda node: len(self.get_neighbors(node)))

    def has_edge(self, node1, node2):
        """判断两个节点是否有边"""
        return node1 in self.edges and node2 in self.edges[node1]

    def copy(self):
        """创建一个图的深拷贝"""
        new_graph = Graph()
        new_graph.nodes = self.nodes.copy()
        new_graph.edges = {node: neighbors.copy() for node, neighbors in self.edges.items()}
        return new_graph

    def degree(self, node):
        """获取节点的度数（邻居数量）"""
        return len(self.get_neighbors(node))

    def subgraph(self, nodes):
        """生成由指定节点集合诱导的子图"""
        subgraph = Graph()
        for node in nodes:
            subgraph.add_node(node)
            for neighbor in self.get_neighbors(node):
                if neighbor in nodes:
                    subgraph.add_edge(node, neighbor)
        return subgraph

    def sort_vertices(self):
        """根据度数对顶点排序（降序）"""
        return sorted(self.nodes, key=lambda x: self.degree(x), reverse=True)

    def to_nx_graph(self):
        """ 将自定义图转换为 NetworkX 图 """
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node)
        for node, neighbors in self.edges.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return G
    def __repr__(self):
        # return Graph information
        return f"Graph(nodes={self.nodes}, edges={self.edges})"

    def parse_dimacs(self, dimacs_file: str):
        """解析DIMACS格式的文件并更新当前Graph对象"""
        with open(dimacs_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('c'):
                    # 跳过注释行
                    continue
                elif line.startswith('p'):
                    # 仅处理 p edge 行，确认文件格式
                    parts = line.split()
                    if len(parts) < 4 or parts[1] != 'edge':
                        raise ValueError("Unsupported graph format.")
                    # 提取节点数和边数（虽然我们不需要使用它们，但可以进行验证等）
                    num_nodes = int(parts[2])
                    num_edges = int(parts[3])
                elif line.startswith('e'):
                    # 处理边
                    parts = line.split()
                    if len(parts) < 3:
                        raise ValueError("Invalid edge format.")
                    node1, node2 = int(parts[1]), int(parts[2])
                    self.add_edge(node1, node2)  # 添加边

    def set_to_adjacency_matrix(graph_set):
        """将集合形式的图转为邻接矩阵存储"""
        # 获取图中所有顶点
        vertices = list(graph_set.keys())
        n = len(vertices)

        # 初始化一个n x n的矩阵，表示图的邻接矩阵
        adjacency_matrix = np.zeros((n, n), dtype=int)

        # 根据邻居集合填充邻接矩阵
        for i, vertex in enumerate(vertices):
            neighbors = graph_set[vertex]
            for neighbor in neighbors:
                j = vertices.index(neighbor)
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1  # 因为是无向图，矩阵是对称的

        return adjacency_matrix
