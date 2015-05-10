# coding=utf-8

import numpy as np
from rdkit import Chem


def smiles2graph(smiles_string, aromatic=0):

    mol = Chem.MolFromSmiles(smiles_string)
    mol_with_h = Chem.AddHs(mol)

    ind_oh_chi_h = []
    atom_with_h = mol_with_h.GetAtoms()
    for i in range(mol_with_h.GetNumAtoms()):
        if atom_with_h[i].GetSymbol() == 'H':
            if int(atom_with_h[i].GetBonds()[0].GetBeginAtom().GetChiralTag()) in (1, 2):
                ind_oh_chi_h.append((atom_with_h[i], atom_with_h[i].GetBonds()[0].GetBeginAtom().GetIdx(),
                                     atom_with_h[i].GetBonds()[0].GetBondType()))

    ind_oh_chi_h.sort(reverse=True)
    mol = Chem.EditableMol(mol)
    for (atom_, i2, bond_type) in ind_oh_chi_h:
        i = mol.AddAtom(atom_)
        mol.AddBond(i2, i, bond_type)
    mol = mol.GetMol()

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
    chi_centers = np.array([int(atoms.GetChiralTag()) for atoms in mol.GetAtoms()])

    num_vertex = np.amax(bond[:, [0, 1]])
    gr = np.zeros((num_vertex, num_vertex))
    hb = np.zeros((num_vertex, num_vertex))

    bond[:, [0, 1]] = bond[:, [0, 1]] - 1
    for i in range(bond.shape[0]):
        gr[bond[i, 0], bond[i, 1]] = bond[i, 2]

    gr = gr + gr.transpose()
    hb = hb + hb.transpose()
    sb = np.zeros((num_vertex, 1))
    for i in range(len(rt)):
        if rt[i] == 1:
            ring_atoms = r['ring' + str(i + 1)]
            ring_index = np.zeros((len(ring_atoms) + 1, 1), dtype=np.int)
            for j in range(len(ring_atoms)):
                ring_index[j] = range(len(an))[np.arange(len(an))[(ring_atoms[j] == an)]]
            ring_index[-1] = ring_index[0]
            sb[ring_index] = 1
            if aromatic:
                ind_i = np.concatenate((ring_index[:-1], ring_index[1:]))
                ind_j = np.concatenate(([ring_index[1:], ring_index[:-1]]))
                mb = np.mean(gr[ind_i.flat, ind_j.flat])
                gr[ind_i.flat, ind_j.flat] = mb

    # TODO what to do with charge?
    gr2 = np.zeros((len(atom), num_h))
    hcount = 0
    for i in range(len(atom)):
        k = nbondrule[i] - nbondcount[i]
        gr2[i, hcount:(hcount + k)] = 1
        hcount += k

    # ap[i,1] - atom's index in smiles, start , ap[i,2] atom's index in smiles, end

    ap = np.zeros((len(atom), 2), dtype=int)
    smiles_string_tmp = smiles_string
    star = '*'
    for (i, atom_) in enumerate(atom):
        ind = smiles_string_tmp.find(atom_)
        ap[i, 0] = ind
        ap[i, 1] = ind + len(atom_) - 1
        smiles_string_tmp = smiles_string_tmp[:ap[i, 0]] + star*len(atom_) + smiles_string_tmp[(ap[i, 1] + 1):]

    for i in range(len(chi_centers)):
        if chi_centers[i] == 1:
            if ap[i, 1] < len(smiles_string):
                if smiles_string[(ap[i, 1]+1):(ap[i, 1]+3)] == '@@':
                    i2 = np.where(ap[:, 0] == ap[i, 1]+3)[0][0]
                    hb[i, i2] = 8
                    hb[i2, i] = 8
        elif chi_centers[i] == 2:
            if ap[i, 1] < len(smiles_string):
                if smiles_string[(ap[i, 1]+1):(ap[i, 1]+2)] == '@':
                    i2 = np.where(ap[:, 0] == ap[i, 1]+2)[0][0]
                    hb[i, i2] = 4
                    hb[i2, i] = 4

    index = np.argsort(ap[:, 0])

    gr = gr[:, index]
    gr = gr[index, :]

    hb = hb[:, index]
    hb = hb[index, :]

    gr2 = gr2[index, :]
    ghs = np.sum(gr2, 1)
    ghs = ghs.astype('int')
    gr2 = np.zeros(np.shape(gr2))
    off = 0L
    for k in range(len(ghs)):
        gr2[k, range(off, (off + ghs[k]))] = 1
        off = off + ghs[k]

    atom = atom[index]
    ap = ap[index, :]
    sb = sb[index]
    charge = charge[index]

    poia = range(np.shape(gr)[0])
    poih = range(np.shape(gr)[0])

    '''
    % gr = основной граф
    % gr2 = граф водорода
    % atom = список атомов (по типам)
    % ap = адреса атомов
    % hb = матрица хиральных связей
    % sb = список атомов, у которых связи будут жесткими
    % charge = заряды атомов
    % poia = доступные точки для инсертов
    % poia = доступные точки для аттачей
    % smiles = исходный smiles
    '''

    return {'g': gr, 'gh': gr2, 'atom': atom, 'atom_pos': ap, 'hb': hb, 'sb': sb, 'charge': charge,
            'poia': poia, 'poih': poih, 'smiles': smiles_string}
