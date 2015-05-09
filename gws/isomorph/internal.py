import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso

def graphs_isomorph_atom(adj1, adj2, atom1, atom2):
    nx_graph1 = mol2nxgraph(adj1, atom1)
    nx_graph2 = mol2nxgraph(adj2, atom2)
    is_iso = nx.faster_could_be_isomorphic(nx_graph1, nx_graph2)
    nm = iso.categorical_node_match('label', 'C')
    em = iso.categorical_edge_match(['weight', 'label'], [1, '-'])
    if is_iso:
        return iso.is_isomorphic(nx_graph1, nx_graph2, node_match=nm, edge_match=em)
    else:
        return is_iso


def mol2nxgraph(adj, atom):

    g1 = nx.Graph()
    n = np.shape(adj)[0]
    for i in range(n):
        g1.add_node(i, label=atom[i])
        for j in range(i+1, n):
            if adj[i, j] > 0:
                if adj[i, j] == 1:
                    lb = '-'
                elif adj[i, j] == 2:
                    lb = '='
                else:
                    lb = '#'
                g1.add_edge(i, j, weight=adj[i, j], label=lb)

    return g1