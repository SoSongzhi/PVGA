import networkx as nx
def is_node_redundant(graph, node):

    for predecessor ,successor in graph.predecessors(node):
        if graph.edges[predecessor, node]['weight'] >= 5 and graph.edges[node, successor]['weight'] >= 5:
            return False
    
    return True




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