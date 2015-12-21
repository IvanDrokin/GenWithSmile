# encoding: utf-8
import re
import numpy as np

from rdkit import Chem

from itertools import izip
from collections import namedtuple
from gws.io.smiles2graph import smiles2graph
from gws.isomorph.symmetric import get_filtered_addons
from gws.star_smiles.parser_exceptions import StarSmilesFormatError
from gws.star_smiles.star_smiles_parser import StarSmilesParser


def star_smiles_to_mol(star_smiles):
    """
    star_smiles: строка в формате star-smiles
    return: граф в виде словаря, описывающего молекулу.
    """
    try:
        parser = StarSmilesParser(star_smiles)
        frame_mol = single_atom_to_graph(parser) or smiles2graph(parser)

        '''
        atom_pos = frame_mol['atom_pos']
        frame_mol['poia'] = _get_interest_atom_indexes(atom_pos, parser.insert_positions)
        frame_mol['poih'] = _get_interest_atom_indexes(atom_pos, parser.attach_positions)
        frame_mol['poia_add'] = np.array([], dtype=int)
        frame_mol['poih_add'] = np.array([], dtype=int)
        frame_mol['history'] = []
        '''
        return frame_mol
    except StopIteration:
        print('Ошибка в star-smiles. Неожиданный конец строки')
    except StarSmilesFormatError as e:
        print(e)
    return None


def data_prep_addons(adds):
    """
    TODO docs
    """
    if 'attach' in adds:
        smiles_appenders = adds['attach']
        names_appenders = adds['names_at']
        if 'two_points' in adds:
            attach_type_flags = adds['two_points']
        else:
            attach_type_flags = [False]*len(smiles_appenders)
    else:
        smiles_appenders, names_appenders, attach_type_flags = [], [], []
    if 'insert' in adds:
        smiles_replaces = adds['insert']
        names_replaces = adds['names_in']
    else:
        smiles_replaces, names_replaces = [], []
    if 'fragment' in adds:
        smiles_fragments = adds['fragment']
        names_fragments = adds['name_fr']
    else:
        smiles_fragments, names_fragments = [], []

    mol_replaces = []
    for smiles, name in izip(smiles_replaces, names_replaces):
        mol = star_smiles_to_mol(smiles)
        mol['name'] = name
        mol_replaces.append(mol)

    mol_appenders = []

    for app_smiles, name, flag in izip(smiles_appenders, names_appenders, attach_type_flags):

        mol = star_smiles_to_mol(app_smiles)
        mol['name'] += name
        mol['attach_type'] = flag
        mol_appenders.append(mol)

    mol_fragments = []
    for app_smiles, name in izip(smiles_fragments, names_fragments):
        mol = star_smiles_to_mol(app_smiles)
        mol['name'] += name
        mol_fragments.append(mol)

    mol_appenders, mol_fragments = get_filtered_addons(mol_appenders, mol_fragments)

    return {
        'insert': mol_replaces,
        'attach': mol_appenders,
        'fragment': mol_fragments
    }


def star_smiles_to_smiles(star_smiles):
    """
    star_smiles: строка в формате star-smiles

    Обозначения формата star-smiles:
      * помечает атом, к которому присоединяется радикал
      ** помечает атом, который заменяется на другой
      *** помечает двумя типами

      Можно помечать группу атомов сразу:
        {...}* или {...}**

      Запись {...} означает, что атомы внутри {} помечены ***.

    return: строку в формате SMILES и списки атомов инсертов и аттачей
    """
    smiles = star_smiles
    attach_atoms = []
    insert_atoms = []

    # паттерны для определения помеченных атомов в порядке их поиска в star-smiles
    # в формате (pattern, type)

    # типы:
    ATTACH = 0b01
    INSERT = 0b10
    ADDON = ATTACH | INSERT

    tokens = [
        ('\*\*\*',      ADDON),   # ***
        ('\{.*?\}\*\*', INSERT),  # {atoms}**
        ('\{.*?\}\*',   ATTACH),  # {atoms}*
        ('\{.*?\}',     ADDON),   # {atoms}
        ('\*\*',        INSERT),  # **
        ('\*',          ATTACH)   # *
    ]

    for pattern, type_ in tokens:
        smiles, ind = token_proc(smiles, pattern, attach_atoms, insert_atoms)
        if type_ & ATTACH:
            attach_atoms += ind
        if type_ & INSERT:
            insert_atoms += ind

    return smiles, insert_atoms, attach_atoms


def token_proc(smiles, pattern, attach_atoms, insert_atoms):
    """
    smiles: строка в формате star-smiles
    pattern: паттерн для разбора формата star-smiles
    attach_atoms: список позиций аттачей
    insert_atoms: список позиций инсертов

    Паттерн может иметь вид (упрощённо) *+ или {...}*+

    return: упрощение строки star_smiles в формате, 
              в котором уже больше не встречается pattern, плюс индексы, 
              атомы на которых следует добавить в соответствующее множество
    """

    simple_case = '\{' not in pattern
    # сдвиг совпадения увеличивается на 2 за счёт скобок {}
    match_shift = pattern.count('\*') + (0 if simple_case else 2)
    indexes = []
    offset = 0

    for match in re.finditer(pattern, smiles):
        real_start = match.start() - offset
        real_end = match.end() - offset

        if simple_case:
            smiles = smiles[:real_start] + smiles[real_end:]
            # Позиция перед началом серии звёзд 
            # должна попадать на конец описания атома
            indexes.append(real_start - 1)
        else:
            # Исключаем {, } и звёзды после }:
            smiles = (smiles[:real_start] + 
                      smiles[real_start + 1 : real_end - match_shift + 1] +  
                      smiles[real_end:])
            # Добавляем позиции, оказавшиеся в {}, с учётом сдвига
            indexes.extend(xrange(real_start, real_end - match_shift))

        offset += match_shift
        _shift_values_greater(attach_atoms, match.end(), -match_shift)
        _shift_values_greater(insert_atoms, match.end(), -match_shift)
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
    atom: атом, для которого генерируется граф

    Если atom совпадает с одним из списка atom_data, 
    то есть возможность быстро преобразовать его в граф

    return: граф, если условия выполнены, иначе возвращает None
    """
    AtomData = namedtuple('AtomData', ['valence'])

    atom_data = {
        'C':  AtomData(valence=4),
        'N':  AtomData(valence=3),
        'Cl': AtomData(valence=1),
        'O':  AtomData(valence=2),
        'I':  AtomData(valence=1),
        'Br':  AtomData(valence=1),
    }

    if atom.smiles not in atom_data:
        return

    bond_multiplexity = {'-': 1, '=': 2, '#': 3}
    mol_data = {
        'poia': np.array([0]),
        'poih': np.array([0]),
        'poif': None,
        'smiles': atom.smiles,
        'aroma_atoms': np.array([0]),
        'rdkit_mol': Chem.MolFromSmiles(atom.smiles)
    }
    if atom.attach_bonds:
        ind, bound = atom.attach_bonds[0]
        mol_data['bound'] = bond_multiplexity[bound]
        mol_data['name'] = '1' + bound
        mol_data['attach_index'] = 0
        mol_data = [mol_data]

    return mol_data
