from eden.graph import Vectorizer
from sklearn import metrics
import networkx as nx
import numpy as np


def graph_kernel(mol_list, sm_list, mol_test, gk_param):
    if len(mol_list) == 0:
        sm_test = []
        for i in range(len(mol_test)):
            sm_test += [mol_test[i]['smiles']]
        return mol_test, sm_test

    p = gk_param['p']
    complexity = gk_param['complexity']
    r = gk_param['r']
    d = gk_param['d']
    min_r = gk_param['min_r']
    min_d = gk_param['min_d']
    nbits = gk_param['nbits']
    rjob = gk_param['rjob']

    graphs_list = pre_process(mol_list)
    graphs_test = pre_process(mol_test)
    vectorizer = Vectorizer(complexity=complexity, r=r, d=d, min_r=min_r, min_d=min_d, nbits=nbits)
    x_list = vectorizer.transform(graphs_list, n_jobs=rjob)
    x_test = vectorizer.transform(graphs_test, n_jobs=rjob)
    k = metrics.pairwise.pairwise_kernels(x_list, x_test, metric='cosine')
    for i in range(len(mol_test)):
        if len(np.where(k[:, i] > p)[0]) < 1:
            mol_list += [mol_test[i]]
            sm_list += [mol_test[i]['smiles']]
    return mol_list, sm_list


def mol2graph(mol):
    adj1 = mol['g']
    atom1 = mol['atom']
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
    a1 = np.concatenate([atom1, mp])

    tmp_zero = np.hstack((b1.transpose(), np.zeros((np.shape(b1)[1], np.shape(b1)[1]))))
    tmp = np.hstack((adj1, b1))
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


def get_def_par():
    return {'p': 0.99999, 'complexity': 4, 'r': None, 'd': None, 'min_r': 0, 'min_d': 0, 'nbits': 20, 'rjob': 1}


def pre_process(iterable):
    for seq in iterable:
        yield mol2graph(seq)
