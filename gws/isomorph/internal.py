# encoding: utf-8
import networkx as nx
import networkx.algorithms.isomorphism as iso


def graphs_isomorph_atom(adj1, adj2, atom1, atom2):
    """
    adj1: матрица смежности первого графа
    adj2: матрица смежности второго графа
    atom1: метки вершин первого графа (символы элементов)
    atom2: метки вершин второго графа (символы элементов)

    return: True если графы изоморфны с учётом типов связей и атомов, иначе False
    """
    nx_graph1 = mol2nxgraph(adj1, atom1)
    nx_graph2 = mol2nxgraph(adj2, atom2)

    is_iso = nx.faster_could_be_isomorphic(nx_graph1, nx_graph2)
    node_match = iso.categorical_node_match('label', 'C')
    edge_match = iso.categorical_edge_match(['weight', 'label'], [1, '-'])
    if is_iso:
        return iso.is_isomorphic(nx_graph1, nx_graph2, 
                                 node_match=node_match, edge_match=edge_match)
    return False


def mol2nxgraph(adjacency_matrix, atom_symbols):
    """
    adjacency_matrix: матрица смежности
    atom_symbols: список имён атомов для меток вершин графа

    return: граф в формате библиотеки networkx, соответствующий переданной 
            матрице смежности, в котором вершинам и рёбрам заданы 
            соответствующие метки
    """
    graph = nx.Graph()
    for i, symbol in enumerate(atom_symbols):
        graph.add_node(i, label=symbol)
    
    edge_type_to_label = {1: '-', 2: '=', 3: '#', 12: '||'}
    n = len(atom_symbols)
    for i in range(n):
        for j in range(i+1, n):
            edge_type = adjacency_matrix[i, j]
            if edge_type == 0:
                continue  # нет ребра

            label = edge_type_to_label.get(edge_type, '')
            graph.add_edge(i, j, weight=edge_type, label=label)

    return graph
