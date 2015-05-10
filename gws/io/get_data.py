# encoding: utf-8
import re
import numpy as np

from itertools import izip
from collections import namedtuple
from gws.io.smiles2graph import smiles2graph


def data_prep_frame(star_smiles):
    """
    star_smiles: строка в формате star_smiles
    Возвращаем граф-словарь, описывающий молекулу:

        g:        основной граф
        gh:       граф водорода
        atom:     список атомов (по типам)
        atom_pos: адреса атомов в smiles: [start end]
        hb:       матрица хиральных связей
        sb:       список атомов, у которых связи будут жесткими
        charge:   массив зарядов атомов
        poia:     доступные точки для инсертов
        poih:     доступные точки для аттачей
        smiles:   исходный smiles
        poia_add:
        poih_add:
        history:
    """
    smiles, replaced_atoms, appended_atoms = star_smiles2smiles(star_smiles)
    frame_mol = single_atom_to_graph(smiles) or smiles2graph(smiles)

    atom_pos = frame_mol['atom_pos']
    
    frame_mol['poia'] = _get_interest_atom_indexes(atom_pos, replaced_atoms)
    frame_mol['poih'] = _get_interest_atom_indexes(atom_pos, appended_atoms)
    frame_mol['poia_add'] = []
    frame_mol['poih_add'] = []
    frame_mol['history'] = []
    return frame_mol


def _get_interest_atom_indexes(all_atom_positions, positions):
    """
    all_atom_positions: массив отрезков [start, end]
    positions: интересующие концы отрезков

    Возвращается массив индексов отрезков, чей конец является ближайшим 
    к какому-то элементу из списка positions
    """
    atom_indexes = np.zeros_like(positions)
    for i, pos in enumerate(positions):
        indexes = np.where(all_atom_positions[:, 1] <= pos)[0]
        if len(indexes) > 0:       # ?? может ли здесь стать True?
            atom_indexes[i] = indexes[-1]
    return np.unique(atom_indexes)


def data_prep_addons(adds):
    """
    TODO docs
    """
    smiles_replaces = adds['insert']
    smiles_appenders = adds['attach']
    names_replaces = adds['names_in']
    names_appenders = adds['names_at']

    mol_replaces = []
    for smiles, name in izip(smiles_replaces, names_replaces):
        mol = data_prep_frame(smiles)
        mol['name'] = name
        mol_replaces.append(mol)

    mol_appenders = []
    for app_smiles, name in izip(smiles_appenders, names_appenders):
        start_bond, smiles = app_smiles[0], app_smiles[1:]
        bond_multiplexity = { '-': 1, '=': 2, '#': 3 }
        mol = data_prep_frame(smiles)
        mol['bound'] = bond_multiplexity[start_bond]
        mol['name'] = name
        mol_appenders.append(mol)

    return {
        'insert': mol_replaces, 
        'attach': mol_appenders
    }


def star_smiles2smiles(star_smiles):
    """
    Принимает на вход строку в формате star_smiles

    * помечает атом, к которому присоединяется радикал
    ** помечает атом, который заменяется на другой
    *** помечает двумя типами

    Можно помечать группу атомов сразу:
    {...}* или {...}**

    Запись {...} означает, что атомы внутри {} помечены ***.

    Возвращает строку smiles и списки атомов для замены и добавления
    """
    smiles = star_smiles
    appended_atoms = []
    replaced_atoms = []

    # паттерны для определения помеченных атомов в порядке их поиска в star_smiles
    # в формате (pattern, type)

    # типы:
    APPEND = 0b01
    REPLACE = 0b10
    BOTH = APPEND | REPLACE

    tokens = [
        ('\*\*\*',      BOTH),    # ***
        ('\{.*?\}\*\*', REPLACE), # {atoms}**
        ('\{.*?\}\*',   APPEND),  # {atoms}*
        ('\{.*?\}',     BOTH),    # {atoms}
        ('\*\*',        REPLACE), # **
        ('\*',          APPEND)   # *
    ]

    for pattern, type_ in tokens:
        smiles, ind = token_proc(smiles, pattern, appended_atoms, replaced_atoms)
        if type_ & APPEND:
            appended_atoms += ind
        if type_ & REPLACE:
            replaced_atoms += ind

    return smiles, replaced_atoms, appended_atoms


def token_proc(smiles, pattern, appended_atoms, replaced_atoms):
    """
    smiles: строка в формате star_smiles
    pattern: паттерн для разбора формата star_smiles

    Возвращает упрощение строки star_smiles в формате, 
    в котором уже больше не встречается pattern, плюс индексы, 
    атомы на которых следует добавить в соответствующее множество

    Паттерн может иметь вид (упрощённо) *+ или {...}*+
    """

    simple_case = '\{' not in pattern
    match_shift = pattern.count('\*') + (0 if simple_case else 2)  # 2 за счёт скобок {}
    indexes = []
    offset = 0

    for match in re.finditer(pattern, smiles):
        real_start = match.start() - offset
        real_end = match.end() - offset

        if simple_case:
            smiles = smiles[:real_start] + smiles[real_end:]
            # Позиция перед началом серии звёзд должна попадать на конец описания атома
            indexes.append(real_start - 1)
        else:
            # Исключаем {, } и звёзды после }:
            smiles = (smiles[:real_start] + 
                      smiles[real_start + 1 : real_end - match_shift + 1] +  
                      smiles[real_end:])
            # Добавляем позиции, оказавшиеся в {}, с учётом сдвига
            indexes.extend(xrange(real_start, real_end - match_shift))

        offset += match_shift
        _shift_values_greater(appended_atoms, match.end(), -match_shift)
        _shift_values_greater(replaced_atoms, match.end(), -match_shift)
    return smiles, indexes


def _shift_values_greater(values, threshold, shift):
    """
    values: список, значения котором будут изменяться
    threshold: пороговое значение
    shift: величина сдвига

    Все значения списка values, большие threshold, будут изменены на shift
    """
    for i, value in enumerate(values):
        if value > threshold:
            values[i] += shift


def single_atom_to_graph(atom):
    """
    Если atom совпадает с одним из списка atom_data, 
    то есть возможность быстро преобразовать его в граф

    Возвращает граф, если условия выполнены, иначе возвращает None
    """
    AtomData = namedtuple('AtomData', ['valence'])

    atom_data = {
        'C':  AtomData(valence=4),
        'N':  AtomData(valence=3),
        'Cl': AtomData(valence=1),
        'O':  AtomData(valence=2),
    }

    if atom not in atom_data:
        return

    data = atom_data[atom]
    mol_data = {
        'g': np.array([[0]]),
        'gh': np.ones((1, data.valence), dtype=int),
        'atom': np.array([atom]),
        'atom_pos': np.array([[0, len(atom) - 1]]),
        'hb': np.array([[0]]),
        'sb': np.array([0]),
        'charge': np.array([0]),
        'poia': np.array([[0]]),
        'poih': np.array([[0]]),
        'smiles': atom
    }

    return mol_data
