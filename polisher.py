import networkx as nx

def is_node_redundant(graph, node):
    for predecessor in graph.predecessors(node):
        if graph.edges[predecessor, node]['weight'] > 2:
            return False
    
    for successor in graph.successors(node):
        if graph.edges[node, successor]['weight'] > 2:
            return False
    
    
    if graph.in_degree(node) == 0 and graph.out_degree(node) >= 2:
        for successor in graph.successors(node):
            if graph.edges[node, successor]['weight'] > 2:
                return False
    
   
    if graph.out_degree(node) == 0 and graph.in_degree(node) >= 2:
        for predecessor in graph.predecessors(node):
            if graph.edges[predecessor, node]['weight'] > 2:
                return False
    
    return True


def polish(graph):
    redundant_nodes = []
    node_labels = {}
    
    for node in graph.nodes:
        if is_node_redundant(graph, node):
            redundant_nodes.append(node)
        else:
            node_labels[node] = graph.nodes[node]['type']
    
    node_types = list(node_labels.values())
    node_types.reverse()
    return graph, [node_types]
