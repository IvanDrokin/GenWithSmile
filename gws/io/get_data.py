# encoding: utf-8
import re
import numpy as np

from gws.io import smiles2graph as s2g


def data_prep_frame(frame_smiles):

    smiles, poia, poih = star_smiles2smiles(frame_smiles)
    frame_mol = check_single_atom(smiles)
    if frame_mol['check'] == -1:
        frame_mol = s2g.smiles2graph(smiles)
    else:
        frame_mol.__delitem__('check')

    atom_pos = frame_mol['atom_pos']
    ind_atoms = np.zeros(len(poia), dtype=int)
    for i in range(len(poia)):
        ind = np.where((atom_pos[:, 1] - poia[i]) <= 0)[0]
        if len(ind) > 0:
            ind_atoms[i] = ind[-1]
    poia = np.unique(ind_atoms)

    ind_atoms = np.zeros(len(poih), dtype=int)
    for i in range(len(poih)):
        ind = np.where((atom_pos[:, 1] - poih[i]) <= 0)[0]
        if len(ind) > 0:
            ind_atoms[i] = ind[-1]
    poih = np.unique(ind_atoms)

    frame_mol['poia'] = poia
    frame_mol['poih'] = poih
    frame_mol['poia_add'] = []
    frame_mol['poih_add'] = []
    frame_mol['history'] = []
    return frame_mol


def data_prep_adds(adds):
    adds_smiles_in = adds['insert']
    adds_smiles_at = adds['attach']
    adds_names_in = adds['names_in']
    adds_names_at = adds['names_at']
    adds_mol_in = []
    for i in range(len(adds_smiles_in)):
        sm = adds_smiles_in[i]
        mol = data_prep_frame(sm)
        mol['name'] = adds_names_in[i]
        adds_mol_in += [mol]

    adds_mol_at = []
    for i in range(len(adds_smiles_at)):
        bound = adds_smiles_at[i][0]
        if bound == '-':
            bound = 1
        elif bound == '-':
            bound = 2
        elif bound == '-':
            bound = 3
        sm = adds_smiles_at[i][1:]
        mol = data_prep_frame(sm)
        mol['bound'] = bound
        mol['name'] = adds_names_at[i]
        adds_mol_at += [mol]

    adds_mol = {'insert': adds_mol_in, 'attach': adds_mol_at}
    return adds_mol


def star_smiles2smiles(star_smiles):
    """
    Принимает на вход строку в формате star_smiles
    возвращает строку smiles и списки атомов для замены и добавления
    """
    smiles = star_smiles
    indh = []
    inda = []

    # паттерны для определения помеченных атомов в порядке их поиска в star_smiles
    # в формате (pattern, is_replace, is_append)
    tokens = [
        ('\*\*\*',                 True,  True),   # ***
        ('\{[a-zA-Z0-9=#]*\}\*\*', False, True),   # {atoms}**
        ('\{[a-zA-Z0-9=#]*\}\*',   True,  False),  # {atoms}*
        # ?? не очень понятно, для чего строка ниже:
        ('\{[a-z|A-Z|0-9|=|#]*\}', True,  True),   # {atoms}
        ('\*\*',                   False, True),   # **
        ('\*',                     True,  False)   # *
    ]

    for pattern, is_replace, is_append in tokens:
        smiles, ind = token_proc(smiles, pattern, inda, indh)
        if is_replace:
            indh += ind
        if is_append:
            inda += ind

    return smiles, inda, indh


def token_proc(smiles, pattern, inda, indh):
    """
    TODO docs
    """
    indexes = []
    offset = 0

    advanced_case = '\{' in pattern

    if not advanced_case:
        for match in re.finditer(pattern, smiles):
            ind_a = match.start() - 1 - offset
            indexes.append(ind_a)
            smiles = smiles[:ind_a + 1] + smiles[match.end() - offset:]
            offset += match.end() - match.start()
       
            for l, p in enumerate(inda):
                if p > match.end():
                    inda[l] -= match.end() - match.start()
            for l, p in enumerate(indh):
                if p > match.end():
                    indh[l] -= match.end() - match.start()
        return smiles, indexes

    num_stars = pattern.count('\*')

    for match in re.finditer(pattern, smiles):
        start_pos = match.start() + 1 - offset
        end_pos = match.end() - 2 - num_stars - offset

        indexes.extend(xrange(start_pos - 1, end_pos))

        smiles = (smiles[:match.start()] +
                  smiles[match.start() + 1 : match.end() - 1 - num_stars] +
                  smiles[match.end() - offset:])

        offset += (2 + num_stars)
        for l, p in enumerate(inda):
            if p > match.end():
                inda[l] -= 2 + num_stars
        for l, p in enumerate(indh):
            if p > match.end():
                indh[l] -= 2 + num_stars
    return smiles, indexes


def check_single_atom(smiles):
    if smiles == 'C':
        mol = {'g': np.zeros((1, 1)), 'gh': np.ones((1, 4)), 'atom': np.array(['C'], dtype='|S1'),
               'atom_pos': np.zeros((1, 2), dtype=int),
               'hb': np.zeros((1, 1)), 'sb': np.array([0]), 'charge': np.array([0]),
               'poia': np.zeros((1, 1), dtype=int),
               'poih': np.zeros((1, 1), dtype=int), 'smiles': 'C', 'check': 1}
    elif smiles == 'N':
        mol = {'g': np.zeros((1, 1)), 'gh': np.ones((1, 3)), 'atom': np.array(['N'], dtype='|S1'),
               'atom_pos': np.zeros((1, 2), dtype=int),
               'hb': np.zeros((1, 1)), 'sb': np.array([0]), 'charge': np.array([0]),
               'poia': np.zeros((1, 1), dtype=int),
               'poih': np.zeros((1, 1), dtype=int), 'smiles': 'N', 'check': 1}
    elif smiles == 'Cl':
        mol = {'g': np.zeros((1, 1)), 'gh': np.ones((1, 1)), 'atom': np.array(['Cl'], dtype='|S1'),
               'atom_pos': np.zeros((1, 2), dtype=int),
               'hb': np.zeros((1, 1)), 'sb': np.array([0]), 'charge': np.array([0]),
               'poia': np.zeros((1, 1), dtype=int),
               'poih': np.zeros((1, 1), dtype=int), 'smiles': 'Cl', 'check': 1}
        mol['atom_pos'][0, 1] = 1
    elif smiles == 'O':
        mol = {'g': np.zeros((1, 1)), 'gh': np.ones((1, 2)), 'atom': np.array(['O'], dtype='|S1'),
               'atom_pos': np.zeros((1, 2), dtype=int),
               'hb': np.zeros((1, 1)), 'sb': np.array([0]), 'charge': np.array([0]),
               'poia': np.zeros((1, 1), dtype=int),
               'poih': np.zeros((1, 1), dtype=int), 'smiles': 'O', 'check': 1}
    else:
        mol = {'check': -1}

    return mol
