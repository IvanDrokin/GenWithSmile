# coding=utf-8

import numpy as np
from rdkit import Chem


def get_smiles_atom_out_order(mol):
    smiles_atom_out_order = mol.GetProp('_smilesAtomOutputOrder')
    # убираем [] и последнюю запятую
    smiles_atom_out_order = map(int, smiles_atom_out_order[1: -2].split(','))
    return smiles_atom_out_order


def smiles2graph(smiles_string):

    mol = Chem.MolFromSmiles(smiles_string)
    smiles_no_chiral = Chem.MolToSmiles(mol, rootedAtAtom=0)

    smiles_atom_out_order = get_smiles_atom_out_order(mol)
    mol_with_h = Chem.AddHs(mol)

    mol_atoms = mol.GetAtoms()
    rings = mol.GetRingInfo()

    # fast hack for cacl parameters with rdkit
    atom = np.array([atom_.GetSymbol() for atom_ in mol_atoms])
    nbondrule = np.array([atom_.GetTotalValence() for atom_ in mol_atoms])
    nbondcount = np.array([atom_.GetExplicitValence() for atom_ in mol_atoms])
    charge = np.array([atom_.GetFormalCharge() for atom_ in mol_atoms])
    an = np.array(['{}_{}'.format(atom_.GetSymbol(), atom_.GetIdx()) for atom_ in mol_atoms])
    num_h = mol_with_h.GetNumAtoms() - mol.GetNumAtoms()
    r = {'ring%s' % i: np.array([an[j] for j in ring]) for i, ring in enumerate(rings.AtomRings())}
    rt = np.array([int(mol.GetBondWithIdx(ring[0]).GetIsAromatic())
                   for ring in rings.BondRings()])
    bond = np.array([[bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1, int(bond.GetBondType())]
                     for bond in mol.GetBonds()])
    chiral_tags = np.array([int(atoms.GetChiralTag()) for atoms in mol.GetAtoms()])

    num_vertex = np.amax(bond[:, [0, 1]])
    gr = np.zeros((num_vertex, num_vertex), dtype=int)

    bond[:, [0, 1]] = bond[:, [0, 1]] - 1
    for i in range(bond.shape[0]):
        gr[bond[i, 0], bond[i, 1]] = bond[i, 2]

    gr = gr + gr.transpose()
    for i in range(len(rt)):
        if rt[i] == 1:
            ring_atoms = r['ring' + str(i)]
            ring_index = np.zeros((len(ring_atoms) + 1, 1), dtype=np.int)
            for j in range(len(ring_atoms)):
                ring_index[j] = range(len(an))[np.arange(len(an))[(ring_atoms[j] == an)]]
            ring_index[-1] = ring_index[0]

    gr2 = np.zeros((len(atom), num_h), dtype=int)
    hcount = 0
    for i in range(len(atom)):
        k = nbondrule[i] - nbondcount[i]
        gr2[i, hcount:(hcount + k)] = 1
        hcount += k

    # ap[i,1] - atom's index in smiles, start , ap[i,2] atom's index in smiles, end

    ap = np.zeros((len(atom), 2), dtype=int)
    smiles_string_tmp = smiles_no_chiral.lower()
    star = '*'
    for i in smiles_atom_out_order:
        atom_ = atom[i]
        ind = smiles_string_tmp.find(atom_.lower())
        ap[i, 0] = ind
        ap[i, 1] = ind + len(atom_) - 1
        smiles_string_tmp = smiles_string_tmp[:ap[i, 0]] + star*len(atom_) + smiles_string_tmp[(ap[i, 1] + 1):]

    index = np.argsort(ap[:, 0])

    gr = gr[:, index]
    gr = gr[index, :]

    gr2 = gr2[index, :]
    ghs = np.sum(gr2, 1)
    gr2 = np.zeros(np.shape(gr2), dtype=int)
    off = 0L
    for k in range(len(ghs)):
        gr2[k, range(off, (off + ghs[k]))] = 1
        off = off + ghs[k]

    atom = atom[index]
    ap = ap[index, :]
    charge = charge[index]

    poia = range(np.shape(gr)[0])
    poih = range(np.shape(gr)[0])

    '''
    % gr = основной граф
    % gr2 = граф водорода
    % atom = список атомов (по типам)
    % ap = адреса атомов
    % chiral_tags = массив типов хиральных центорв для каждого атома, соответствует Chem.rdchem.ChiralType
    % charge = заряды атомов
    % poia = доступные точки для инсертов
    % poia = доступные точки для аттачей
    % smiles = исходный smiles без информации о изомерах
    '''

    return {'g': gr, 'gh': gr2, 'atom': atom, 'atom_pos': ap, 'chiral_tags': chiral_tags,
            'charge': charge, 'poia': poia, 'poih': poih, 'smiles': smiles_no_chiral}
