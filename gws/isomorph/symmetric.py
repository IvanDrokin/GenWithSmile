# coding=utf-8
from contextlib import contextmanager

import numpy as np

from gws.isomorph.internal import is_isomorph_nx, mol2nxgraph


def get_symmetric(mol, attach_positions=None, insert_positions=None):
    """
    mol: объект класса gws.Molecule, представляющий собой молекулу
    attach_positions: список позиций в графе, подлежащий проверке изоморфизма аттача
    insert_positions: список позиций в графе, подлежащий проверке изоморфизма инсерта

    return: новый список позиций для аттача и для инсерта, в которых оставлены 
            только представители класса эквивалентности по соответствующему изоморфизму
    """
    if attach_positions is None:
        attach_positions = []
    if insert_positions is None:
        insert_positions = []

    graph = mol2nxgraph(mol.g, mol.atom)

    attach_positions = _get_nonisomorphic_positions(graph, add_vertex_to, attach_positions)
    insert_positions = _get_nonisomorphic_positions(graph, change_vertex, insert_positions)

    return attach_positions, insert_positions


def _get_nonisomorphic_positions(graph, editor_manager, positions):
    """
    graph: граф в формате networkx, в котором требуется проверить изменения на изоморфность
    editor_manager: contextmanager, который изменяет вершину в графе. 
    positions: массив позиций, изменения в которых требуется проверить на изоморфность

    return: список позиций, при изменении вершин которых 
            получится множество неизоморфных графов
    """
    is_isomorph = np.zeros_like(positions, dtype=bool)
    for (_, graph1), (j, graph2) in _get_graph_combinations(graph, editor_manager, 
                                                            positions, is_isomorph):
        is_isomorph[j] = is_isomorph_nx(graph1, graph2)

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
