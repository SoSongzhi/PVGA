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

   
    start_node = next(iter(graph.nodes))
    dp[start_node] = 0

    
    for node in graph:
        
        for neighbor in graph.successors(node):
            weight = graph.edges[node, neighbor]['weight']
            
            if dp[node] + weight > dp[neighbor]:
                dp[neighbor] = dp[node] + weight
                path[neighbor] = node

   
    end_node = max(dp, key=dp.get)
    max_weight = dp[end_node]

    
    current_node = end_node
    while current_node is not None:
        if dp[current_node] == 0:
            start_node = current_node
            break
        current_node = path[current_node]


    max_weight_path = []
    node_labels = [[]]
    current_node = end_node

    while current_node is not None:
        max_weight_path.append(current_node)
        node_labels[0].append(node_types[current_node])
        current_node = path[current_node]
    max_weight_path.reverse()
    node_labels.reverse()

   
    max_weight_graph = nx.DiGraph()
    for node in max_weight_path:
        node_type = graph.nodes[node].get('type')  
        max_weight_graph.add_node(node, type=node_type)  
        if path[node] is not None:
            weight = graph.edges[path[node], node].get('weight')  
            max_weight_graph.add_edge(path[node], node, weight=weight)

    return max_weight, max_weight_path, node_labels, max_weight_graph


def max_weight_path_old(graph, start, end):
    dp = {node: float('-inf') for node in graph}
    dp[start] = 0  
    path = {node: None for node in graph}
    node_types = {node: graph.nodes[node]['type'] for node in graph}

    
    for node in graph:
        
        for neighbor in graph.successors(node):
            weight = graph.edges[node, neighbor]['weight']
            if dp[node] + weight > dp[neighbor]:
                dp[neighbor] = dp[node] + weight
                path[neighbor] = node

    
    max_weight_path = []
    node_labels = [[]]
    current_node = end
    while current_node is not None:
        max_weight_path.append(current_node)
        node_labels[0].append(node_types[current_node])
        current_node = path[current_node]
    max_weight_path.reverse()
    node_labels.reverse()
    max_weight_graph = nx.DiGraph()
    for node in max_weight_path:
        node_type = graph.nodes[node].get('type')  
        max_weight_graph.add_node(node, type=node_type)  
        if path[node] is not None:
            weight = graph.edges[path[node], node].get('weight')  
            max_weight_graph.add_edge(path[node], node, weight=weight)

    return dp[end], max_weight_path, node_labels, max_weight_graph

# choose paths with max ratio
def max_ratio_paths(graph, start, end):
    dp = {node: float('-inf') for node in graph}
    dp[start] = 0  
    paths = {node: [] for node in graph}  
    paths[start] = [[start]] 

    node_types = {node: graph.nodes[node]['type'] for node in graph}  
    path_counts = {node: 0 for node in graph} 

    
    for node in graph:
        
        for neighbor in graph.successors(node):
            weight = graph.edges[node, neighbor]['weight']
            ratio = (dp[node] + weight) / (path_counts[node] + 1)
            if path_counts[neighbor] == 0 or ratio > dp[neighbor] / path_counts[neighbor]:
                dp[neighbor] = dp[node] + weight
                paths[neighbor] = [path + [neighbor] for path in paths[node]]
                path_counts[neighbor] = path_counts[node] + 1
           
            elif ratio == dp[neighbor] / path_counts[neighbor]:
                paths[neighbor].extend([path + [neighbor] for path in paths[node]])
                path_counts[neighbor] += path_counts[node]
    max_paths = paths[end]
    node_labels = [[node_types[node] for node in path] for path in max_paths]

    return dp[end], max_paths, node_labels, path_counts[end]
