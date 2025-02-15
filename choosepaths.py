import networkx as nx
import numpy as np
from numba import cuda
import torch
cuda.select_device(0)
cuda.get_current_device().reset()

from numba import config
config.CUDA_IR_VERSION = 1.6

def is_node_redundant(graph, node):

    for predecessor ,successor in graph.predecessors(node):
        if graph.edges[predecessor, node]['weight'] >= 5 and graph.edges[node, successor]['weight'] >= 5:
            return False
    
    return True

def max_weight_path_gpu(graph, start, end):
    # 获取图中所有节点和边
    nodes = list(graph.nodes)
    edges = list(graph.edges(data='weight'))
    node_indices = {node: i for i, node in enumerate(nodes)}

    # 将节点的类型映射到整数类别编码
    node_type_mapping = {node_type: i for i, node_type in enumerate(
        {graph.nodes[node]['type'] for node in nodes}
    )}
    node_types = torch.tensor(
        [node_type_mapping[graph.nodes[node]['type']] for node in nodes],
        dtype=torch.int32
    ).to('cuda')

    # 初始化 DP 表和路径表
    dp = torch.full((len(nodes),), float('-inf'), device='cuda')  # 在 GPU 上
    dp[node_indices[start]] = 0
    path = torch.full((len(nodes),), -1, dtype=torch.int32, device='cuda')

    # 将边数据转换为张量
    edge_indices = torch.tensor(
        [[node_indices[u], node_indices[v]] for u, v, _ in edges], device='cuda'
    )
    edge_weights = torch.tensor(
        [float(data) if data is not None else 0.0 for _, _, data in edges],
        dtype=torch.float32, device='cuda'
    )

    # 更新 DP 表
    for _ in range(len(nodes)):
        for i, (u, v) in enumerate(edge_indices):
            weight = edge_weights[i]
            if dp[u] + weight > dp[v]:
                dp[v] = dp[u] + weight
                path[v] = u

    # 构建最大权值路径
    max_weight_path = []
    node_labels = []
    current_index = node_indices[end]
    while current_index != -1:
        node = nodes[current_index]
        max_weight_path.append(node)
        node_labels.append(node_types[current_index].item())
        current_index = path[current_index].item()

    max_weight_path.reverse()
    node_labels.reverse()

    # 构建最大权值图
    max_weight_graph = nx.DiGraph()
    for i in range(len(max_weight_path)):
        node = max_weight_path[i]
        node_type = graph.nodes[node]['type']
        max_weight_graph.add_node(node, type=node_type)
        if i > 0:
            prev_node = max_weight_path[i - 1]
            weight = graph.edges[prev_node, node]['weight']
            max_weight_graph.add_edge(prev_node, node, weight=weight)

    return dp[node_indices[end]].item(), max_weight_path, node_labels, max_weight_graph
def max_weight_path(graph):
    dp = {node: float('-inf') for node in graph}
    path = {node: None for node in graph}
    node_types = {node: graph.nodes[node]['type'] for node in graph}

    # 随意选择一个节点作为起点并设置其dp值为0
    start_node = next(iter(graph.nodes))
    dp[start_node] = 0

    # 遍历每个节点
    for node in graph:
        # 遍历节点的出边
        for neighbor in graph.successors(node):
            weight = graph.edges[node, neighbor]['weight']
            # 更新最大权值和
            if dp[node] + weight > dp[neighbor]:
                dp[neighbor] = dp[node] + weight
                path[neighbor] = node

    # 找到dp值最大的点作为end节点
    end_node = max(dp, key=dp.get)
    max_weight = dp[end_node]

    # 回溯找到起点start_node，即dp值为0的节点
    current_node = end_node
    while current_node is not None:
        if dp[current_node] == 0:
            start_node = current_node
            break
        current_node = path[current_node]

    # 构建最大权值路径
    max_weight_path = []
    node_labels = [[]]
    current_node = end_node

    while current_node is not None:
        max_weight_path.append(current_node)
        node_labels[0].append(node_types[current_node])
        current_node = path[current_node]
    max_weight_path.reverse()
    node_labels.reverse()

    # 构建最大权值图
    max_weight_graph = nx.DiGraph()
    for node in max_weight_path:
        node_type = graph.nodes[node].get('type')  # 获取节点的 'type' 属性
        max_weight_graph.add_node(node, type=node_type)  # 添加节点及其属性
        if path[node] is not None:
            weight = graph.edges[path[node], node].get('weight')  # 获取边的权值
            max_weight_graph.add_edge(path[node], node, weight=weight)

    return max_weight, max_weight_path, node_labels, max_weight_graph


def max_weight_path_old(graph, start, end):
    dp = {node: float('-inf') for node in graph}
    dp[start] = 0  # 起点权值设为0
    path = {node: None for node in graph}
    node_types = {node: graph.nodes[node]['type'] for node in graph}

    # 遍历每个节点
    for node in graph:
        # 遍历节点的出边
        for neighbor in graph.successors(node):
            weight = graph.edges[node, neighbor]['weight']
            # 更新最大权值和
            if dp[node] + weight > dp[neighbor]:
                dp[neighbor] = dp[node] + weight
                path[neighbor] = node

    # 构建最大权值路径
    max_weight_path = []
    node_labels = [[]]
    current_node = end
    while current_node is not None:
        max_weight_path.append(current_node)
        node_labels[0].append(node_types[current_node])
        current_node = path[current_node]
    max_weight_path.reverse()
    node_labels.reverse()


    # 从graph当中找到最大path返回graph格式的最大path

    # 构建最大权值图
    max_weight_graph = nx.DiGraph()
    for node in max_weight_path:
        node_type = graph.nodes[node].get('type')  # 获取节点的 'type' 属性
        max_weight_graph.add_node(node, type=node_type)  # 添加节点及其属性
        if path[node] is not None:
            weight = graph.edges[path[node], node].get('weight')  # 获取边的权值
            max_weight_graph.add_edge(path[node], node, weight=weight)

    return dp[end], max_weight_path, node_labels, max_weight_graph

# choose paths with max ratio
def max_ratio_paths(graph, start, end):
    dp = {node: float('-inf') for node in graph}
    dp[start] = 0  # Set the weight of the start node to 0
    paths = {node: [] for node in graph}  # Store path lists
    paths[start] = [[start]]  # Initialize the path for the start node

    node_types = {node: graph.nodes[node]['type'] for node in graph}  # Add node type information
    path_counts = {node: 0 for node in graph}  # Store the number of paths from start to each node

    # Traverse each node
    for node in graph:
        # Traverse the outgoing edges of the node
        for neighbor in graph.successors(node):
            weight = graph.edges[node, neighbor]['weight']
            # Calculate the weight-to-path ratio for the current edge
            ratio = (dp[node] + weight) / (path_counts[node] + 1)
            # Update the maximum weight and paths based on the ratio
            if path_counts[neighbor] == 0 or ratio > dp[neighbor] / path_counts[neighbor]:
                dp[neighbor] = dp[node] + weight
                paths[neighbor] = [path + [neighbor] for path in paths[node]]
                path_counts[neighbor] = path_counts[node] + 1
            # If there are equal maximum ratios, add paths to the existing path list
            elif ratio == dp[neighbor] / path_counts[neighbor]:
                paths[neighbor].extend([path + [neighbor] for path in paths[node]])
                path_counts[neighbor] += path_counts[node]
    max_paths = paths[end]
    # Get the node labels for the maximum path
    node_labels = [[node_types[node] for node in path] for path in max_paths]

    return dp[end], max_paths, node_labels, path_counts[end]




# Kernel 函数：并行更新 dp 和 path
@cuda.jit
def update_dp_path(dp, path, weights, successors, nodes_per_level, level_offset):
    idx = cuda.grid(1)
    if idx >= nodes_per_level:
        return

    # 当前节点和偏移
    current_node = level_offset + idx
    for j in range(successors[current_node][0]):  # successors 是邻接列表
        neighbor = successors[current_node][j + 1]
        weight = weights[current_node][j]
        
        # 使用原子操作更新 dp 和 path
        cuda.atomic.max(dp, neighbor, dp[current_node] + weight)
        if dp[neighbor] == dp[current_node] + weight:
            path[neighbor] = current_node

# 主函数：将数据传输到 GPU 并启动 Kernel
def max_weight_path_cuda(graph):
    # 建立节点到索引的映射
    node_to_index = {node: idx for idx, node in enumerate(graph.nodes())}
    index_to_node = {idx: node for node, idx in node_to_index.items()}

    # 计算 max_degree 和 start_node
    max_degree = max(len(list(graph.neighbors(node))) for node in graph)
    start_node = next(node for node in graph if graph.in_degree(node) == 0)
    start_node_index = node_to_index[start_node]

    # 图数据转化为 GPU 格式
    n_nodes = len(graph)
    dp = np.full(n_nodes, -np.inf, dtype=np.float32)
    path = np.full(n_nodes, -1, dtype=np.int32)

    weights = np.zeros((n_nodes, max_degree), dtype=np.float32)
    successors = np.zeros((n_nodes, max_degree + 1), dtype=np.int32)

    # 填充邻接表和权重
    for node in graph:
        node_idx = node_to_index[node]
        neighbors = list(graph.neighbors(node))
        successors[node_idx, 0] = len(neighbors)
        for j, neighbor in enumerate(neighbors):
            neighbor_idx = node_to_index[neighbor]
            successors[node_idx, j + 1] = neighbor_idx
            weights[node_idx, j] = graph.edges[node, neighbor]['weight']

    # 初始化 dp[start_node_index] 为 0
    dp[start_node_index] = 0

    # 将数据复制到 GPU
    d_dp = cuda.to_device(dp)
    d_path = cuda.to_device(path)
    d_weights = cuda.to_device(weights)
    d_successors = cuda.to_device(successors)

    # 配置 CUDA grid 和 block
    threads_per_block = 256
    blocks_per_grid = (n_nodes + threads_per_block - 1) // threads_per_block

    # 假设 levels 是图的层级信息
    levels = [(0, n_nodes)]
    for level_offset, nodes_per_level in levels:
        update_dp_path[blocks_per_grid, threads_per_block](
            d_dp, d_path, d_weights, d_successors, nodes_per_level, level_offset
        )

    # 结果返回到主机
    dp = d_dp.copy_to_host()
    path = d_path.copy_to_host()

    # 找到最大 dp 和路径
    end_node_index = np.argmax(dp)
    max_weight_path = []
    while end_node_index != -1:
        max_weight_path.append(index_to_node[end_node_index])
        end_node_index = path[end_node_index]
    max_weight_path.reverse()

    return max_weight_path, dp[np.argmax(dp)]