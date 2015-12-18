# coding=utf-8

import numpy as np
from rdkit import Chem


def get_smiles_atom_out_order(mol):
    smiles_atom_out_order = mol.GetProp('_smilesAtomOutputOrder')
    # убираем [] и последнюю запятую
    smiles_atom_out_order = map(int, smiles_atom_out_order[1: -2].split(','))
    return smiles_atom_out_order


def smiles2graph(star_smiles_parser):

    mol = Chem.MolFromSmiles(star_smiles_parser.smiles)
    Chem.Kekulize(mol)
    smiles_no_chiral = Chem.MolToSmiles(mol, rootedAtAtom=0)
    smiles_atom_out_order = get_smiles_atom_out_order(mol)
    mol_atoms = mol.GetAtoms()

    # fast hack for cacl parameters with rdkit
    atom = np.array([atom_.GetSymbol() for atom_ in mol_atoms])

    # ap[i,1] - atom's index in smiles, start , ap[i,2] atom's index in smiles, end
    ap = np.zeros((len(atom), 2), dtype=int)
    smiles_string_tmp = smiles_no_chiral.lower()
    star = '*'
    for i in smiles_atom_out_order:
        atom_ = atom[i]
        ind = smiles_string_tmp.find(atom_.lower())
        ap[i, 0] = ind
        ap[i, 1] = ind + len(atom_) - 1
        smiles_string_tmp = smiles_string_tmp[:ap[i, 0]] + star*len(atom_) + \
            smiles_string_tmp[(ap[i, 1] + 1):]

    if star_smiles_parser.attach_bonds:
        inds, bonds = zip(*star_smiles_parser.attach_bonds)
        atom_index = _get_interest_atom_indexes(ap, inds,)
        poia = _get_interest_atom_indexes(ap, star_smiles_parser.insert_positions)
        poih = _get_interest_atom_indexes(ap, star_smiles_parser.attach_positions)

        bond_multiplexity = {'-': 1, '=': 2, '#': 3}
        bonds = [bond_multiplexity[i] for i in bonds]
        new_molec = {'poia': poia, 'poih': poih,
                     'smiles': star_smiles_parser.smiles,
                     'poia_add': np.array([], dtype=int), 'poih_add': np.array([], dtype=int),
                     'history': [], 'rdkit_mol': mol,
                     'attach_index': atom_index, 'bond': bonds,
                     'name': '', 'poif': None}

        return new_molec

    ap = np.zeros((len(atom), 2), dtype=int)
    smiles_string_tmp = smiles_no_chiral.lower()
    star = '*'
    for i in smiles_atom_out_order:
        atom_ = atom[i]
        ind = smiles_string_tmp.find(atom_.lower())
        ap[i, 0] = ind
        ap[i, 1] = ind + len(atom_) - 1
        smiles_string_tmp = smiles_string_tmp[:ap[i, 0]] + star*len(atom_) + \
            smiles_string_tmp[(ap[i, 1] + 1):]

    index = np.argsort(ap[:, 0])

    ap = ap[index, :]

    poia = _get_interest_atom_indexes(ap, star_smiles_parser.insert_positions)
    poih = _get_interest_atom_indexes(ap, star_smiles_parser.attach_positions)
    poif = _get_interest_atom_indexes(ap, star_smiles_parser.fragment_pos)

    return {'poia': poia, 'poih': poih, 'smiles': smiles_no_chiral,
            'poia_add': np.array([], dtype=int), 'poih_add': np.array([], dtype=int),
            'history': [], 'rdkit_mol': mol, 'name': '', 'poif': poif}


def _get_interest_atom_indexes(all_atom_positions, positions):
    """
    all_atom_positions: массив отрезков [start, end]
    positions: интересующие концы отрезков

    return: массив индексов отрезков, чей конец является ближайшим
              к какому-то элементу из списка positions
    """
    atom_indexes = np.zeros_like(positions)
    for i, pos in enumerate(positions):
        indexes = np.where(all_atom_positions[:, 1] <= pos)[0]
        if len(indexes) > 0:       # ?? может ли здесь стать True?
            atom_indexes[i] = indexes[-1]
    return np.unique(atom_indexes)
