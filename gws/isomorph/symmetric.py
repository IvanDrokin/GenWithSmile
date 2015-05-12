import numpy as np

from gws.isomorph import internal


def get_symmetric(mol, poih=None, poia=None):
    # {'g': gr, 'gh': gr2, 'atom': atom, 'atom_pos': ap, 'hb': hb, 'sb': sb, 'charge': charge}
    if poih is None:
        poih = []
    if poia is None:
        poia = []

    g = mol['g'] + mol['hb']
    a = mol['atom']

    n = np.shape(g)[0]
    marks = np.ones_like(poih)

    for i in range(len(poih) - 1):
        if marks[i] != 1:
            continue
        for j in range(i + 1, len(poih)):
            g1 = g.copy()
            g1 = np.hstack([g1, np.zeros((np.shape(g1)[0], 1))])
            g1 = np.vstack([g1, np.zeros((1, np.shape(g1)[1]))])
            g1[n, poih[i]] = 1
            g1[poih[i], n] = 1
            a1 = a.copy()
            a1 = np.append(a1, 'mh')
            g2 = g.copy()
            g2 = np.hstack([g2, np.zeros((np.shape(g2)[0], 1))])
            g2 = np.vstack([g2, np.zeros((1, np.shape(g2)[1]))])
            g2[poih[j], n] = 1
            g2[n, poih[j]] = 1
            a2 = a.copy()
            a2 = np.append(a2, 'mh')
            isomorph = internal.graphs_isomorph_atom(g1, g2, a1, a2)
            if isomorph:
                marks[j] = 0
    invh = poih[np.where(marks == 1)]

    indk = poia
    k = len(indk)
    marks = np.ones((np.shape(indk)))

    for i in range(k-1):
        if marks[i] == 1:
            for j in range(i+1, k):
                g1 = g.copy()
                a1 = a.copy()
                a1[indk[i]] = 'mh'
                g2 = g.copy()
                a2 = a.copy()
                a2[indk[j]] = 'mh'
                isomorph = internal.graphs_isomorph_atom(g1, g2, a1, a2)
                if isomorph:
                    marks[j] = 0
    inva = indk[np.where(marks == 1)]
    return invh, inva
