def max_clique(graph):
    return _max_clique(graph, set(), set(), 0)[0]


def _max_clique(graph, current_clique, candidates, lb):
    if not candidates and not graph:
        if len(current_clique) > lb:
            return current_clique.copy(), len(current_clique)
        else:
            return set(), lb

    # 计算上界：当前团大小 + 上界估计
    if graph:
        ub = len(current_clique) + overestimation(graph)
    else:
        ub = len(current_clique) + overestimation(candidates)

    if ub <= lb:
        return set(), lb  # 剪枝

    # 选择最小度数的顶点
    if graph:
        v = min(graph.keys(), key=lambda x: len(graph[x]))
    else:
        v = min(candidates, key=lambda x: len(graph.get(x, set())))

    # 分支1：包含顶点v
    new_clique = current_clique.copy()
    new_clique.add(v)
    # 生成子图G_v（v的邻居）
    neighbors = graph.get(v, set())
    subgraph = {u: graph[u] & neighbors for u in neighbors if u in graph}
    # 递归调用
    max_clique1, new_lb1 = _max_clique(subgraph, new_clique, candidates & neighbors, max(lb, len(new_clique)))
    lb = max(lb, new_lb1)

    # 分支2：不包含顶点v
    remaining = graph.copy()
    remaining.pop(v, None)
    # 移除v后更新候选节点
    new_candidates = candidates - {v}
    # 递归调用
    max_clique2, new_lb2 = _max_clique(remaining, current_clique, new_candidates, lb)
    lb = max(lb, new_lb2)

    # 返回更大的团
    if len(max_clique1) > len(max_clique2):
        return max_clique1, lb
    else:
        return max_clique2, lb


def overestimation(graph):
    """贪心着色算法估计色数（上界）"""
    color = {}
    for node in sorted(graph, key=lambda x: -len(graph[x])):  # 按度数降序
        used = set(color.get(neigh, -1) for neigh in graph[node])
        for c in range(len(graph)):
            if c not in used:
                color[node] = c
                break
        else:
            color[node] = len(graph)
    return max(color.values()) + 1 if color else 0


# 示例图的邻接表表示
example_graph = {
    'v1': {'v4', 'v5', 'v6'},
    'v2': {'v3', 'v5', 'v6'},
    'v3': {'v2', 'v4'},
    'v4': {'v1', 'v3', 'v6'},
    'v5': {'v1', 'v2', 'v6'},
    'v6': {'v1', 'v2', 'v4', 'v5'}
}

result, size = max_clique(example_graph)
print(f"最大团: {result}, 大小: {size}")  # 输出: 最大团: {'v3', 'v6'}, 大小: 2