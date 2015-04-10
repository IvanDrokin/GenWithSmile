import numpy as np
import igraph as ig


def graphs_isomorph_atom(adj1, adj2, atom1, atom2):
    g1 = ig.Graph()
    g1.is_weighted = 1
    g2 = ig.Graph()
    g2.is_weighted = 1

    b1 = np.zeros((np.shape(adj1)[0], len(atom1)))
    c1 = 0
    mp = np.zeros((len(atom1)), dtype='object')
    mp[0] = atom1[0]
    b1[0, 0] = 1
    for i in range(1, len(atom1)):
        ind = np.where(mp[0:(c1+1)] == atom1[i])[0]
        if len(ind) > 0:
            b1[i, ind] = 1
        else:
            c1 += 1
            mp[c1] = atom1[i]
            b1[i, c1] = 1
    b1 = b1[:, 0:(c1+1)]

    b2 = np.zeros((np.shape(adj2)[0], len(atom2)))
    c2 = 0
    b2[0, 0] = 1
    for i in range(1, len(atom2)):
        ind = np.where(mp[0:(c2+1)] == atom2[i])[0]
        if len(ind) > 0:
            b2[i, ind] = 1
        else:
            c2 += 1
            mp[c2] = atom2[i]
            b2[i, c2] = 1
    b2 = b2[:, 0:(c2+1)]

    tmp_zero = np.hstack((b1.transpose(), np.zeros((np.shape(b1)[1], np.shape(b1)[1]))))
    tmp = np.hstack((adj1, b1))
    a1b = np.vstack((tmp, tmp_zero))
    tmp_zero = np.hstack((b2.transpose(), np.zeros((np.shape(b2)[1], np.shape(b2)[1]))))
    tmp = np.hstack((adj2, b2))
    a2b = np.vstack((tmp, tmp_zero))

    numver = np.sum(a1b)/2
    numver = numver.astype('int')
    g1.add_vertices(numver)

    l = []
    n = np.shape(a1b)[0]
    for i in range(n):
        for j in range(i+1, n):
            if a1b[i, j] > 0:
                l.append((i, j))
                l.append((j, i))
    g1.add_edges(l)

    numver = np.sum(a2b)/2
    numver = numver.astype('int')
    g2.add_vertices(numver)
    l = []
    n = np.shape(a2b)[0]
    for i in range(n):
        for j in range(i+1, n):
            if a2b[i, j] > 0:
                l.append((i, j))
                l.append((j, i))
    g2.add_edges(l)

    isomorph = g1.isomorphic(g2)
    return isomorph
