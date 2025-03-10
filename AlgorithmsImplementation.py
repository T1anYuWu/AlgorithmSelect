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

    """max_clq start"""

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
    def kclique(self,graph, P, k):
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
                return True, clique # 找到团，返回节点集合

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

    def initialize_k_value(self,graph):
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
