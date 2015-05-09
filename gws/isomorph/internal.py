import numpy as np
import networkx as nx


def graphs_isomorph_atom(adj1, adj2, atom1, atom2):
    nx_graph1 = mol2nxgraph(adj1, atom1)
    nx_graph2 = mol2nxgraph(adj2, atom2)
    is_iso = nx.faster_could_be_isomorphic(nx_graph1, nx_graph2)
    if is_iso:
        return nx.is_isomorphic(nx_graph1, nx_graph2)
    else:
        return is_iso


def mol2nxgraph(adj, atom):

    b1 = np.zeros((np.shape(adj)[0], len(atom)))
    c1 = 0
    mp = np.zeros((len(atom)), dtype='object')
    mp[0] = atom[0]
    b1[0, 0] = 1
    for i in range(1, len(atom)):
        ind = np.where(mp[0:(c1+1)] == atom[i])[0]
        if len(ind) > 0:
            b1[i, ind] = 1
        else:
            c1 += 1
            mp[c1] = atom[i]
            b1[i, c1] = 1
    b1 = b1[:, 0:(c1+1)]
    a1 = np.concatenate([atom, mp])

    tmp_zero = np.hstack((b1.transpose(), np.zeros((np.shape(b1)[1], np.shape(b1)[1]))))
    tmp = np.hstack((adj, b1))
    a1b = np.vstack((tmp, tmp_zero))

    g1 = nx.Graph()
    n = np.shape(a1b)[0]
    for i in range(n):
        g1.add_node(i, label='')
        for j in range(i+1, n):
            if a1b[i, j] > 0:
                g1.add_edge(i, j, weight=a1b[i, j], label='-')
    for i in range(n):
        g1.node[i]['lable'] = a1[i]

    return g1