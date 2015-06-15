import numpy as np
from eden.graph import Vectorizer
from sklearn import metrics
from scipy.sparse import vstack
from internal import mol2nxgraph


def graph_kernel(mol_list, mol_test, gk_param):
    vectorizer = Vectorizer(complexity=gk_param['complexity'], r=gk_param['r'], d=gk_param['d'],
                            min_r=gk_param['min_r'], min_d=gk_param['min_d'],
                            nbits=gk_param['nbits'])
    graphs_list = (mol2nxgraph(mol.g, mol.atom) for mol in mol_test)

    if not mol_list:
        mol_feat_list = vectorizer.transform(graphs_list)
        for (i, mol) in enumerate(mol_test):
            mol.graph_kernel_vect = mol_feat_list[i, :]
        return mol_test

    mol_feat_list = vstack([mol.graph_kernel_vect for mol in mol_list])
    x_test = vectorizer.transform(graphs_list)
    k = metrics.pairwise.pairwise_kernels(mol_feat_list, x_test, metric='cosine')
    for i, mol in enumerate(mol_test):
        if len(np.where(k[:, i] > gk_param['p'])[0]) < 1:
            mol.graph_kernel_vect = x_test[i, :]
            mol_list.append(mol)
    return mol_list


def get_def_par():
    return {'p': 0.99999, 'complexity': 4, 'r': None, 'd': None, 'min_r': 0,
            'min_d': 0, 'nbits': 20, 'rjob': 1}
