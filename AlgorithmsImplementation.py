from copy import deepcopy
from itertools import combinations
import networkx as nx
from pysat.examples.rc2 import RC2
from pysat.solvers import Glucose3, Solver
import GraphAndFileProcess
import numpy as np
from pysat.formula import CNF, WCNF
from pysat.solvers import Glucose3
import sys


#sys.setrecursionlimit(2000)


class GraphProcessor:
    def __init__(self, graph: GraphAndFileProcess.Graph):
        self.graph = graph


class max_clique:
    def __init__(self, graph: GraphAndFileProcess.Graph):
        self.graph = graph

    def run(self):
        return self.max_clq(self.graph, C=None, LB=0)

    def overestimation(self, graph):
        """计算图 G 的最大团的过估计上限"""
        P = []  # 存储独立集的集合
        remaining_graph = graph.copy()  # 深拷贝图

        # 主循环，直到图为空
        while remaining_graph.nodes:
            # 步骤4：选择度数最大的节点 v
            v = max(remaining_graph.nodes, key=lambda node: remaining_graph.degree(node))
            remaining_graph.remove_node(v)  # 步骤5：移除节点 v

            # 创建独立集
            S = {v}  # 新独立集初始只包含 v

            # 步骤6：添加与 v 邻接的其他兼容节点
            for neighbor in graph.get_neighbors(v):
                # 验证这个邻居不能与 S 中的任何节点相连
                if all(neighbor not in graph.get_neighbors(n) for n in S):
                    S.add(neighbor)  # 仅在与 S 中的节点无连接时添加

            P.append(S)  # 添加新独立集

        #print(f"Independent sets formed: {P}")  # 输出生成的独立集

        # 使用 WCNF 形式编码 MaxSAT
        wcnf = WCNF()

        # 添加硬子句：每对非相连节点形成硬子句
        nodes = list(graph.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if not graph.has_edge(u, v):
                    wcnf.append([-u, -v])  # 添加硬子句

        # 将所有独立集添加为软子句
        for S in P:
            if len(S) > 1:
                wcnf.append([-literal for literal in S])  # 形成软子句

        # 统计失败字面量
        s = 0  # 初始化失效计数器
        failed_literals = set()

        for clause in wcnf.soft:
            if all(literal in failed_literals for literal in clause):
                s += 1  # 统计未满足的失败字面量
            else:
                failed_literals.update(clause)  # 更新失败字面量集合

        return len(P) - s  # 返回独立集数量减去失败的数量

    def max_clq(self, graph, C=None, LB=0):
        """分支限界算法查找最大团"""
        if C is None:
            C = set()  # 初始化当前团

        # 如果没有节点，返回当前团
        if len(graph.nodes) == 0:
            return C

        # 计算上界
        UB = self.overestimation(graph) + len(C)
        # print(f"Calculating upper bound: {UB}")

        # 剪枝条件
        if LB >= UB:
            return set()

        # 选择度数最小的节点
        v = graph.select_min_degree_vertex()
        # print(f"Selected vertex: {v}")

        # 包含 v 的情况下的递归调用
        C1 = self.max_clq(graph.subgraph(graph.get_neighbors(v)), C | {v}, LB)

        # 更新下界
        if len(C1) > LB:
            LB = len(C1)
            # print(f"Updated LB to: {LB}")

        # 不包含 v 的情况下的递归调用
        G_not_v = graph.copy()
        G_not_v.remove_node(v)
        C2 = self.max_clq(G_not_v, C, LB)

        # 返回较大的团
        max_clique = C1 if len(C1) >= len(C2) else C2

        # 输出找到的团及其大小
        if max_clique:
            print(f"Found clique: {max_clique}, Size: {len(max_clique)}")

        return max_clique

    """max_clq end"""

    """k-clique start"""


class k_clique:
    def __init__(self, graph: GraphAndFileProcess.Graph):
        self.graph = graph

    def run(self, k_max=None):
        """执行 k-clique 算法"""
        # 如果没有传入 k_max，使用默认的初始化 K 值
        if k_max is None:
            k_max = self.initialize_k_value(self.graph)
        # 运行 k-clique 计算并返回结果
        return self.kclique_sequence(self.graph, k_max)

    def kclique(self, graph, P, k):
        """
        递归判断是否存在大小为 k 的团
        graph: 图对象
        P: 当前候选节点集合
        k: 目标团大小
        """
        if k == 1:
            if not P:  # 如果 P 为空，返回 False
                return False, set()
            return True, {next(iter(P))}  # 返回任意一个单节点作为团

        KCCNS = self.construct_kclique_covering_node_set(graph, P, k)

        for p in KCCNS:
            new_P = P.intersection(graph.get_neighbors(p))
            found, clique = self.kclique(graph, new_P, k - 1)
            if found:
                clique |= {p}
                #print(f"找到一个大小为 {k} 的团，顶点集合为: {clique}")
                return True, clique  # 找到团，返回节点集合

            P.remove(p)  # 移除当前节点，继续尝试

        return False, set()  # 没找到，返回空集合

    def kclique_sequence(self, graph, k_max):
        """
        graph: 图对象
        k_max: 初始最大团估计值
        """
        FOUND = False
        k = k_max
        max_clique = set()
        print(f"初始化 K 值为 {k}")

        while not FOUND and k >= 1:
            P = set(graph.nodes)  # V = 所有节点
            print(f"正在查找大小为 {k} 的团")
            FOUND, max_clique = self.kclique(graph, P, k)  # 递归查找 k-团

            if not FOUND:
                print(f"未找到大小为 {k} 的团，正在递减 k 值")
                k -= 1  # 继续递减 k
        print(f"最大团的顶点集合为: {max_clique}")
        return max_clique  # 返回最大团的节点集合

    def construct_kclique_covering_node_set(self, graph, P, k):
        """
        构造 k-团覆盖节点集（基于节点度数选取高可能性节点）
        """
        return sorted(P, key=lambda node: len(graph.get_neighbors(node)), reverse=True)[:k]

    def initialize_k_value(self, graph):
        """
        使用图着色算法初始化 K 值，返回色数作为 K 的上界
        """
        color_map = {}
        for node in graph.nodes:
            # 获取当前节点相邻的已着色节点
            neighbor_colors = {color_map[n] for n in graph.get_neighbors(node) if n in color_map}

            # 为节点分配一个最小的未使用颜色
            color_map[node] = min(set(range(len(graph.nodes))) - neighbor_colors)

        # 色数就是 color_map 中最大颜色的编号 + 1
        chromatic_number = max(color_map.values()) + 1
        return chromatic_number

    """k-clique end"""


class DBK_Algorithm:
    def __init__(self, graph: GraphAndFileProcess.Graph):
        """
        初始化DBK算法类

        参数:
        graph: 需要处理的图，networkx 图实例
        """
        self.graph = graph.to_nx_graph()

    def run(self,LIMIT=None):
        if LIMIT is None:
            return self.DBK(LIMIT=50, solver_function=self.maximum_clique_exact_solve_np_hard)
        return self.DBK(LIMIT, solver_function=self.maximum_clique_exact_solve_np_hard)

    def maximum_clique_exact_solve_np_hard(self, graph):
        """
        计算最大团

        输入：
        - G：自定义图，形式为 Graph(nodes={...}, edges={...})

        输出：
        - 最大团，返回一个列表，包含最大团的节点
        """
        # 将自定义图转换为 NetworkX 图


        # 使用 NetworkX 的 find_cliques 函数找到所有团
        cliques = list(nx.find_cliques(graph))  # 找到所有团
        max_clique = max(cliques, key=len)  # 获取最大团
        return max_clique
    def mc_upper_bound(self, G):
        """
        计算最大团的上界

        输入:
        - G: Networkx 图

        输出:
        - 上界（色数）
        """
        answ = nx.algorithms.coloring.greedy_color(G)
        chromatic_number = list(set(list(answ.values())))
        return len(chromatic_number)

    def mc_lower_bound(self, G):
        """
        计算最大团的下界

        输入:
        - G: Networkx 图

        输出:
        - 下界（最大独立集的补图）
        """
        return nx.maximal_independent_set(nx.complement(G))

    def edge_k_core(self, G, k):
        """
        计算图的k-core

        输入:
        - G: Networkx 图
        - k: k-core的阈值

        输出:
        - G: k-core简化后的图
        """
        for a in list(G.edges()):
            x = list(G.neighbors(a[0]))
            y = list(G.neighbors(a[1]))
            if len(list(set(x) & set(y))) <= (k-2):
                G.remove_edge(a[0], a[1])
        return G

    def k_core_reduction(self, graph, k):
        """
        计算图的k-core简化

        输入:
        - graph: Networkx 图
        - k: k-core的阈值

        输出:
        - graph: k-core简化后的图
        """
        graph = nx.k_core(graph, k)
        ref1 = len(list(graph.edges()))
        graph = self.edge_k_core(graph, k)
        ref2 = len(list(graph.edges()))
        while ref1 != ref2:
            if len(graph) == 0:
                return graph
            graph = nx.k_core(graph, k)
            ref1 = len(list(graph.edges()))
            graph = self.edge_k_core(graph, k)
            ref2 = len(list(graph.edges()))
        return graph

    def is_clique(self, G):
        """
        判断一个子图是否为团

        输入:
        - G: Networkx 图

        输出:
        - True 如果是团，False 如果不是团
        """
        n = len(list(G.nodes()))
        m = len(list(G.edges()))
        return m == (n * (n - 1)) / 2

    def ch_partitioning(self, vertex, G):
        """
        图分割操作

        输入:
        - vertex: 分割的节点
        - G: Networkx 图

        输出:
        - SSG: 左子图
        - SG: 右子图
        """
        n = list(G.neighbors(vertex))
        Gp = []
        for iter in list(G.edges()):
            if iter[0] in n and iter[1] in n:
                Gp.append(iter)
        G.remove_node(vertex)
        return nx.Graph(Gp), G

    def lowest_degree_vertex(self, graph):
        """
        查找度数最小的节点

        输入:
        - graph: Networkx 图

        输出:
        - 度数最小的节点
        """
        degrees = [graph.degree(a) for a in list(graph.nodes())]
        minimum = min(degrees)
        for i in list(graph.nodes()):
            if graph.degree(i) == minimum:
                return i

    def remove_zero_degree_nodes(self, graph):
        """
        移除度数为0的节点

        输入:
        - graph: Networkx 图

        输出:
        - graph: 去除度数为0节点后的图
        """
        nodes = list(graph.nodes())
        for n in nodes:
            if graph.degree(n) == 0:
                graph.remove_node(n)
        return graph

    def DBK(self, LIMIT, solver_function):
        """
        DBK算法实现

        输入:
        - LIMIT: 最大图大小，超过该限制的图会被递归处理
        - solver_function: 求解器函数，用于求解最大团

        输出:
        - k: 求得的最大团
        """
        assert type(self.graph) is nx.Graph
        assert type(LIMIT) is int
        assert len(self.graph) != 0

        G = self.graph.copy()
        if len(self.graph) <= LIMIT:
            return solver_function(self.graph)

        self.graph = self.remove_zero_degree_nodes(self.graph)
        k = self.mc_lower_bound(self.graph)
        self.graph = self.k_core_reduction(self.graph, len(k))

        if len(self.graph) == 0:
            return k
        if len(self.graph) <= LIMIT:
            return solver_function(self.graph)

        vertex_removal = {self.graph: []}
        subgraphs = [self.graph]
        while subgraphs:
            SG = subgraphs.pop()
            SG = self.remove_zero_degree_nodes(SG)
            assert len(SG) != 0
            vertex = self.lowest_degree_vertex(SG)
            SSG, SG = self.ch_partitioning(vertex, SG)
            SG = self.remove_zero_degree_nodes(SG)
            SSG = self.remove_zero_degree_nodes(SSG)

            if self.is_clique(SG) and len(SG) > len(k):
                k = list(SG.nodes())

            if len(SSG) != 0:
                SSG_lower = self.mc_lower_bound(SSG)
                if len(SSG_lower) > len(k):
                    k = SSG_lower
                else:
                    subgraphs.append(SSG)

            if len(SG) != 0:
                SG_lower = self.mc_lower_bound(SG)
                if len(SG_lower) > len(k):
                    k = SG_lower
                else:
                    subgraphs.append(SG)
        return k



class CliSAT:
    pass
    """CliSat end"""

class bro_kerbosch_algorithm:
    def __init__(self, graph: GraphAndFileProcess.Graph):
        self.graph = graph
    def run(self):
        max_clique=self.bron_kerbosch()
        return max_clique
    def bron_kerbosch(self):
        """实现 Bron-Kerbosch 算法，寻找最大团"""
        R = set()  # 当前团
        P = set(self.graph.nodes)  # 未处理节点集
        X = set()  # 已处理节点集
        cliques = []

        def bron_kerbosch_recursive(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            for node in list(P):
                bron_kerbosch_recursive(R.union({node}), P.intersection(self.graph.get_neighbors(node)),
                                        X.intersection(self.graph.get_neighbors(node)))
                P.remove(node)
                X.add(node)

        bron_kerbosch_recursive(R, P, X)
        return max(cliques, key=len) if cliques else set()
