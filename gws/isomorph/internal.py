# coding=utf-8
from rdkit import Chem

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
    return is_isomorph_nx(mol2nxgraph(adj1, atom1), mol2nxgraph(adj2, atom2))

    
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


def get_list_of_smiles(mol_tmp):
    smiles = []
    for mol in mol_tmp:
        rdkit_mol = Chem.MolFromSmiles(mol.smiles)
        chiral_type_by_index = {0: Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                                1: Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                                2: Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                                3: Chem.rdchem.ChiralType.CHI_OTHER}
        for (j, atom) in enumerate(rdkit_mol.GetAtoms()):
            atom.SetChiralTag(chiral_type_by_index.get(mol.chiral_tags[j]))
        smiles.append(Chem.MolToSmiles(rdkit_mol, rootedAtAtom=0, isomericSmiles=True))
    return smiles