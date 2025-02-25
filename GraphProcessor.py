import Graph


class GraphProcessor:
    def __init__(self, graph: Graph):
        self.graph = graph

    def max_clq(self):
        pass
    def bron_kerbosch(self):
        """实现Bron-Kerbosch算法，寻找最大团"""
        R = set()  # 当前团
        P = set(self.graph.nodes)  # 未处理节点集
        X = set()  # 已处理节点集
        cliques = []

        def bron_kerbosch_recursive(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            for node in list(P):
                bron_kerbosch_recursive(R.union({node}),P.intersection(self.graph.get_neighbors(node)),X.intersection(self.graph.get_neighbors(node)))
                P.remove(node)
                X.add(node)

        bron_kerbosch_recursive(R, P, X)
        return max(cliques, key=len) if cliques else set()
