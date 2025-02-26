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
        # get one node neighbors
        return self.edges.get(node, set())

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