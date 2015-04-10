import numpy as np
import copy

from gws.isomorph import internal


def get_symmetric(mol, poih=copy.copy([]), poia=copy.copy([])):
    # {'g': gr, 'gh': gr2, 'atom': atom, 'atom_pos': ap, 'hb': hb, 'sb': sb, 'charge': charge}
    g = mol['g'] + mol['hb']
    #gh = mol['gh']
    a = mol['atom']
    '''
    if len(poih) == 0:
        indk = np.arange(0, np.shape(gh)[0], dtype='int')
        indk = indk[np.where(np.sum(gh, 1) > 0)]
    else:
        indk = poih '''
    indk = poih
    n = np.shape(g)[0]
    k = len(indk)
    marks = np.ones(np.shape(indk))

    for i in range((k-1)):
        if marks[i] == 1:
            for j in range(i+1, k):
                g1 = g.copy()
                g1 = np.hstack([g1, np.zeros((np.shape(g1)[0], 1))])
                g1 = np.vstack([g1, np.zeros((1, np.shape(g1)[1]))])
                g1[n, indk[i]] = 1
                g1[indk[i], n] = 1
                a1 = a.copy()
                a1 = np.append(a1, 'mh')
                g2 = g.copy()
                g2 = np.hstack([g2, np.zeros((np.shape(g2)[0], 1))])
                g2 = np.vstack([g2, np.zeros((1, np.shape(g2)[1]))])
                g2[indk[j], n] = 1
                g2[n, indk[j]] = 1
                a2 = a.copy()
                a2 = np.append(a2, 'mh')
                isomorph = internal.graphs_isomorph_atom(g1, g2, a1, a2)
                if isomorph:
                    marks[j] = 0
    invh = indk[np.where(marks == 1)]

    '''
    if len(poia) == 0:
        indk = np.arange(0, np.shape(g)[1], dtype='int')
    else:
        indk = poia '''
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
