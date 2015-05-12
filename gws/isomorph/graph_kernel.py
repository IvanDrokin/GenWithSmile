from eden.graph import Vectorizer
from sklearn import metrics
from scipy.sparse import vstack
from internal import mol2nxgraph
import numpy as np


def graph_kernel(mol_list, sm_list, mol_feat_list, mol_test, gk_param):
    p = gk_param['p']
    complexity = gk_param['complexity']
    r = gk_param['r']
    d = gk_param['d']
    min_r = gk_param['min_r']
    min_d = gk_param['min_d']
    nbits = gk_param['nbits']
    rjob = gk_param['rjob']

    vectorizer = Vectorizer(complexity=complexity, r=r, d=d, min_r=min_r, min_d=min_d, nbits=nbits)
    graphs_list = (mol2nxgraph(mol['g'], mol['atom']) for mol in mol_test)

    if not mol_list:
        mol_feat_list = vectorizer.transform(graphs_list, n_jobs=rjob)
        sm_test = [mol['smiles'] for mol in mol_test]
        return mol_test, sm_test, mol_feat_list

    x_test = vectorizer.transform(graphs_list, n_jobs=rjob)
    k = metrics.pairwise.pairwise_kernels(mol_feat_list, x_test, metric='cosine')
    for i, mol in enumerate(mol_test):
        if len(np.where(k[:, i] > p)[0]) < 1:
            mol_list.append(mol)
            sm_list.append(mol['smiles'])
            mol_feat_list = vstack([mol_feat_list, x_test[i, ]])
    return mol_list, sm_list, mol_feat_list


def get_def_par():
    return {'p': 0.99999, 'complexity': 4, 'r': None, 'd': None, 'min_r': 0, 'min_d': 0, 'nbits': 20, 'rjob': 1}
