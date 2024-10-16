import networkx as nx
def read_graph_from_file(filename):
    graph = nx.DiGraph()
    with open(filename, 'r') as file:
        lines = file.readlines()

        # Obtain the num of v adn e
        line = lines[0].strip().split('\t')
        num_vertices = int(line[1])
        num_edges = int(line[3])

        # read the v

        for line in lines[1 :1 +  num_vertices]:
            line = line.strip().split('\t')
            vertex_id = line[1]

            vertex_type = line[2]
            graph.add_node(vertex_id, type=vertex_type)

        # read the e
        for line in lines[1 + num_vertices:]:
            line = line.strip().split('\t')
            edge_type = line[0]
            start_vertex = line[1]
            end_vertex = line[2]
            weight = int(line[3].split(':')[1]) # 修改这一行
            graph.add_edge(start_vertex, end_vertex, weight=weight)

    return graph