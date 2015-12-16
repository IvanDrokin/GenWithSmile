# coding=utf-8
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np


def is_isomorph_nx(graph1, graph2):
    """
    graph1, graph2: графы в формате networkx, изоморфность которых проверяется
    return: True, если графы изоморфны, иначе False
    """
    is_iso = nx.faster_could_be_isomorphic(graph1, graph2)
    node_match = iso.categorical_node_match('label', 'C')
    edge_match = iso.categorical_edge_match(['weight', 'label'], [1, '-'])
    if is_iso:
        return iso.is_isomorphic(graph1, graph2,
                                 node_match=node_match, edge_match=edge_match)
    return False


def get_list_of_smiles(mol_tmp):
    smiles = []
    for mol in mol_tmp:
        smiles.append(mol.smiles)
    return smiles


def rdkitmol2graph(mol):
    """
    Преобразование молекулы в граф
    :param mol: rdkit.Chem.rdchem.Mol
    :return: networkx.Graph
    """
    mol_atoms = mol.GetAtoms()

    rings = mol.GetRingInfo()
    num_rings = len(rings.BondRings())
    # fast hack for cacl parameters with rdkit
    charge = np.array([atom_.GetFormalCharge() for atom_ in mol_atoms] + [0]*num_rings)
    atom = np.array([atom_.GetSymbol() for atom_ in mol_atoms] + ['R']*num_rings)
    bond = np.array([[bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1, int(bond.GetBondType())]
                     for bond in mol.GetBonds()])
    num_vertex = np.amax(bond[:, [0, 1]])
    gr = np.zeros((num_vertex+num_rings, num_vertex+num_rings), dtype=int)

    bond[:, [0, 1]] = bond[:, [0, 1]] - 1
    for i in range(bond.shape[0]):
        gr[bond[i, 0], bond[i, 1]] = bond[i, 2]

    gr = gr + gr.transpose()

    for k, ring in enumerate(rings.AtomRings()):
        gr[ring, num_vertex + k] = 13
        gr[num_vertex + k, ring] = 13

    graph = nx.Graph()
    for i, symbol in enumerate(atom):
        graph.add_node(i, label=symbol, entity=charge[i])

    edge_type_to_label = {1: '-', 2: '=', 3: '#', 12: '||', 13: 'RING'}
    n = len(atom)
    for i in range(n):
        for j in range(i+1, n):
            edge_type = gr[i, j]
            if edge_type == 0:
                continue  # нет ребра

            label = edge_type_to_label.get(edge_type, '')
            graph.add_edge(i, j, weight=edge_type, label=label)

    return graph
