# coding=utf-8
from itertools import product
from contextlib import contextmanager

import numpy as np

from gws.isomorph.internal import is_isomorph_nx, rdkitmol2graph


def get_symmetric(mol, attach_positions=None, insert_positions=None, fragment_points=None):
    """
    mol: объект класса gws.Molecule, представляющий собой молекулу
    attach_positions: список позиций в графе, подлежащий проверке изоморфизма аттача
    insert_positions: список позиций в графе, подлежащий проверке изоморфизма инсерта

    return: новый список позиций для аттача и для инсерта, в которых оставлены
            только представители класса эквивалентности по соответствующему изоморфизму
    """
    if attach_positions is None:
        attach_positions = np.asarray([])
    if insert_positions is None:
        insert_positions = np.asarray([])
    if fragment_points is None:
        fragment_points = np.asarray([])

    graph = rdkitmol2graph(mol.rdkit_mol)

    attach_positions = _get_nonisomorphic_positions(graph, add_vertex_to, attach_positions)
    insert_positions = _get_nonisomorphic_positions(graph, change_vertex, insert_positions)
    fragment_points = _get_nonisomorphic_positions(graph, change_vertex, fragment_points)

    # filter two-point attach
    node_index, edje_w, node_label = [], [], []
    for i in attach_positions.tolist():
        for j in attach_positions.tolist():
            if i != j:
                node_index.append([i, j])
                edje_w.append([1, 1])
                node_label.append(['LABEL_ONE', 'LABEL_TWO'])
    positions = _get_nonisomorphic_positions(graph, add_vertexes,
                                             zip(node_index, node_label, edje_w))
    two_p_attach_positions = [node_index[i] for i in positions]

    return attach_positions, insert_positions, fragment_points, two_p_attach_positions


def get_filtered_addons(mol_appenders, mol_fragments):
    mol_appenders_out = []
    mol_fragments_out = []
    # filter one-point attach
    for mol in mol_appenders:
        if len(mol['attach_index']) == 1:
            mol_copy = mol.copy()
            mol_copy['attach_type'] = False
            mol_appenders_out.append(mol_copy)
            continue
        mol_copy = mol.copy()
        graph = rdkitmol2graph(mol['rdkit_mol'])
        node_index = mol['attach_index']
        edje_w = mol['bond']
        node_label = ['LABEL']*len(edje_w)
        positions = _get_nonisomorphic_positions(graph, add_vertexes,
                                                 zip(node_index, node_label, edje_w))
        mol_copy['bond'] = [edje_w[i] for i in positions]
        mol_copy['attach_index'] = [node_index[i] for i in positions]
        mol_copy['attach_type'] = False
        mol_appenders_out.append(mol_copy)

    # filter two-point attach
    for mol in mol_appenders:
        if len(mol['attach_index']) == 1 or not mol['attach_type']:
            continue
        mol_copy = mol.copy()
        graph = rdkitmol2graph(mol['rdkit_mol'])
        node_index, edje_w, node_label = [], [], []
        for i in xrange(len(mol['attach_index'])):
            for j in xrange(len(mol['attach_index'])):
                node_index.append([mol['attach_index'][i], mol['attach_index'][j]])
                edje_w.append([mol['bond'][i], mol['bond'][j]])
                node_label.append(['LABEL_ONE', 'LABEL_TWO'])
        positions = _get_nonisomorphic_positions(graph, add_vertexes,
                                                 zip(node_index, node_label, edje_w))
        mol_copy['bond'] = [edje_w[i] for i in positions]
        mol_copy['attach_index'] = [node_index[i] for i in positions]
        mol_copy['attach_type'] = True
        mol_appenders_out.append(mol_copy)

    # filter fragments
    _int_to_str = {1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR'}
    for mol in mol_fragments:
        mol_copy = mol.copy()
        graph = rdkitmol2graph(mol['rdkit_mol'])
        atom_idexes = [i for i in xrange(len(mol['rdkit_mol'].GetAtoms()))]

        # now generate sets of points for 2, 3 and 4 branches
        points = {}
        cur_arom_index = [atom_idexes]
        allowed_bond_mult = [2, 3]
        for k in allowed_bond_mult:
            bonds = ['BOND_' + _int_to_str[i] for i in xrange(1, k+1)]
            label = ['LABEL_' + _int_to_str[i] for i in xrange(1, k+1)]
            node_index, edje_w, node_label = [], [], []
            for _a in product(*(cur_arom_index + [atom_idexes])):
                ind_list = [_a[i] for i in xrange(k)]
                if len(set(ind_list)) == len(ind_list):
                    node_index.append(ind_list)
                    edje_w.append(bonds)
                    node_label.append(label)
            positions = _get_nonisomorphic_positions(graph, add_vertexes,
                                                     zip(node_index, node_label, edje_w))
            points[k] = {}
            points[k]['index'] = [node_index[i] for i in positions]
            cur_arom_index = points[k]['index']
        mol_copy['points'] = points
        mol_fragments_out.append(mol_copy)
    return mol_appenders_out, mol_fragments_out


def _get_nonisomorphic_positions(graph, editor_manager, positions):
    """
    graph: граф в формате networkx, в котором требуется проверить изменения на изоморфность
    editor_manager: contextmanager, который изменяет вершину в графе.
    positions: массив позиций, изменения в которых требуется проверить на изоморфность

    return: список позиций, при изменении вершин которых
            получится множество неизоморфных графов
    """
    if type(positions) is list:
        is_isomorph = np.zeros(len(positions), dtype=bool)
    else:
        is_isomorph = np.zeros_like(positions, dtype=bool)
    for (_, graph1), (j, graph2) in _get_graph_combinations(graph, editor_manager,
                                                            positions, is_isomorph):
        is_isomorph[j] = is_isomorph_nx(graph1, graph2)
    if type(positions) is list:
        return np.where(~is_isomorph)[0].tolist()
    else:
        return positions[~is_isomorph]


def _get_graph_combinations(graph, editor_manager, positions, no_edit_mask):
    """
    graph: граф в формате networkx
    editor_manager: contextmanager, который управляет изменением графа.
                    Должен принимать на вход граф и id изменяемой вершины
    positions: номера вершин графов, подлежащие изменению
    no_edit_mask: позиции в списке positions, пары с которыми не требуют обработки.
                  Может изменяться внешним кодом во время итерации

    return: последовательность всех комбинаций изменённых графов.
            Возвращённая комбинация состоит из двух кортежей, в каждом из которых
            по два элемента: номер позиции из списка positions, в котором произведено
            изменение и изменённый соответствующим образом граф
    """
    graph1, graph2 = graph, graph.copy()
    for i in range(len(positions) - 1):
        if no_edit_mask[i]:
            continue
        with editor_manager(graph1, positions[i]):
            for j in range(i + 1, len(positions)):
                if no_edit_mask[j]:
                    continue
                with editor_manager(graph2, positions[j]):
                    yield (i, graph1), (j, graph2)


@contextmanager
def add_vertex_to(graph, node_id):
    """
    graph: граф, в котором будет изменена вершина
    node_id: id вершины, к которой будет добавлена фиктивная

    Контекстный менеджер, безопасно добавляющий и удаляющий уникальную вершину
      к графу.

    return: граф с добавленной к node_id уникальной вершиной
    """
    SYMMETRIC_LABEL = 'symmetric_label'
    SYMMETRIC_WEIGHT = -1
    added_node_id = len(graph)
    graph.add_node(added_node_id, label=SYMMETRIC_LABEL)
    graph.add_edge(node_id, added_node_id,
                   weight=SYMMETRIC_WEIGHT, label=SYMMETRIC_LABEL)
    yield
    graph.remove_node(added_node_id)


@contextmanager
def add_vertexes(graph, position):
    node_ids, node_labels, edge_labels = position
    if type(node_ids) is not list:
        node_ids = [node_ids]
        node_labels = [node_labels]
        edge_labels = [edge_labels]
    added_node_ids = []
    for k, (node_is, node_l, edge_l) in enumerate(zip(node_ids, node_labels, edge_labels)):
        new_node_index = len(graph) + k
        graph.add_node(new_node_index, label=node_l)
        graph.add_edge(node_is, new_node_index,
                       weight=edge_l, label=str(edge_l))
        added_node_ids.append(new_node_index)
    yield
    for added_node_id in added_node_ids:
        graph.remove_node(added_node_id)


@contextmanager
def change_vertex(graph, node_id):
    """
    graph: граф, в котором будет изменена вершина
    node_id: id вершины, которая будет заменена на уникальную

    Контекстный менеджер, безопасно изменяющий и восстанавливающий вершину node_id.

    return: граф с добавленной к node_id уникальной вершиной
    """
    SYMMETRIC_LABEL = 'symmetric_label'
    stored_label = graph.node[node_id]['label']
    graph.node[node_id]['label'] = SYMMETRIC_LABEL
    yield
    graph.node[node_id]['label'] = stored_label
