# coding=utf-8
import itertools
import numpy as np

from rdkit import Chem

from gws import Molecule

from gws.isomorph import graph_kernel as gk
from gws.isomorph import symmetric, internal


def generate(num_iter, m0, adds, is_test=0, gk_param=gk.get_def_par()):
    #  {'g': gr, 'gh': gr2, 'atom': atom, 'atom_pos': ap, 'hb': hb, 'sb': sb,
    #  'charge': charge, 'poih': poih, 'poia': poia}
    poih = m0.poih
    poia = m0.poia
    list_mols = gk.graph_kernel([], [m0], gk_param)
    invh, inva = symmetric.get_symmetric(m0, poih, poia)
    list_mols_gen1 = generate_1(m0, invh, inva, adds)
    list_mols = gk.graph_kernel(list_mols, list_mols_gen1, gk_param)
    curr_m = list_mols[1:]
    for i in range(1, num_iter):
        # nl_m = []
        nl_m_len = len(list_mols)
        for molecule in curr_m:
            poih = molecule.poih
            poia = molecule.poia
            # Check symmetric
            invh, inva = symmetric.get_symmetric(molecule, poih, poia)
            # generate_1
            mols = generate_1(molecule, invh, inva, adds)
            # cut
            if is_test == 0:
                if len(mols) > 0:
                    list_mols = gk.graph_kernel(list_mols, mols, gk_param)
            else:
                list_mols = unconcat(list_mols, mols)
        curr_m = list_mols[nl_m_len:-1]
    list_mols = list_mols[1:]
    list_mols_smiles = internal.get_list_of_smiles(list_mols)
    for i in range(len(list_mols)):
        list_mols[i].poia = list_mols[i].poia_add
        list_mols[i].poih = list_mols[i].poih_add
        list_mols[i].poih_add = np.array([], dtype=int)
        list_mols[i].poia_add = np.array([], dtype=int)
    return list_mols, list_mols_smiles


def unconcat(nl_m, new_mols):
    mask = np.ones((np.shape(new_mols)[0], 1))
    for (i, new_mol) in enumerate(new_mols):
        for j in nl_m:
            somorph = graphs_isomorph_mol(new_mol, j)
            if somorph > 0:
                mask[i] = 0
                break
    indx = np.where(mask == 1)[0]
    new_nonisomorphic_mols = [new_mols[i] for i in indx]
    nl_m += new_nonisomorphic_mols
    return nl_m


def graphs_isomorph_mol(mol1, mol2):
    g1 = internal.rdkitmol2graph(mol1.rdkit_mol)
    g2 = internal.rdkitmol2graph(mol2.rdkit_mol)
    return internal.is_isomorph_nx(g1, g2)


def generate_1(mol, invh, inva, adds_in):

    generated_mols = []
    for i in invh:
        generated_mols += get_mols_with_attachs(i, mol, adds_in['attach'])

    for i in inva:
        generated_mols += get_mols_with_inserts(i, mol, adds_in['insert'])

    return generated_mols


def get_mols_with_attachs(target_point, mol0, addons):

    rdkit_mol0 = mol0.rdkit_mol
    poia0 = mol0.poia
    poih0 = mol0.poih
    poia_add0 = mol0.poia_add
    poih_add0 = mol0.poih_add
    hist0 = mol0.history

    mols_with_attachs = []

    target_free_b = rdkit_mol0.GetAtomWithIdx(target_point).GetTotalValence() - \
        rdkit_mol0.GetAtomWithIdx(target_point).GetExplicitValence()

    for attach in addons:
        mh_ad = attach['bound']
        mh_index = attach['attach_index']
        target_free_at = attach['rdkit_mol'].GetAtomWithIdx(mh_index).GetTotalValence() - \
            attach['rdkit_mol'].GetAtomWithIdx(mh_index).GetExplicitValence()
        if target_free_b >= mh_ad and mh_ad <= target_free_at:
            new_rdkitmol = rdkit_mol0.__copy__()
            poia = poia0.copy()
            poih = poih0.copy()
            poia_add = poia_add0.copy()
            poih_add = poih_add0.copy()
            hist = list(hist0)

            em = Chem.EditableMol(new_rdkitmol)

            num_atom1 = len(new_rdkitmol.GetAtoms())

            for atom in attach['rdkit_mol'].GetAtoms():
                em.AddAtom(atom)

            for bond in attach['rdkit_mol'].GetBonds():
                em.AddBond(bond.GetBeginAtomIdx() + num_atom1, bond.GetEndAtomIdx() + num_atom1,
                           bond.GetBondType())
            bond = Chem.rdchem.BondType.SINGLE
            if mh_ad == 1:
                bond = Chem.rdchem.BondType.SINGLE
            elif mh_ad == 2:
                bond = Chem.rdchem.BondType.DOUBLE
            elif mh_ad == 3:
                bond = Chem.rdchem.BondType.TRIPLE
            elif mh_ad == 4:
                bond = Chem.rdchem.BondType.QUADRUPLE
            elif mh_ad == 5:
                bond = Chem.rdchem.BondType.QUINTUPLE
            em.AddBond(target_point, mh_index + num_atom1, bond)

            outmol = em.GetMol()
            Chem.SanitizeMol(outmol)
            poia_add = np.hstack([poia_add, num_atom1 + attach['poia'].copy()])
            poih_add = np.hstack([poih_add, num_atom1 + attach['poih'].copy()])
            hist.append('a-' + str(target_point) + '-' + attach['name'])
            mol_out = Molecule({'poia': poia, 'poih': poih, 'poia_add': poia_add,
                                'poih_add': poih_add, 'smiles': Chem.MolToSmiles(outmol),
                                'history': hist, 'rdkit_mol': outmol})
            mols_with_attachs.append(mol_out)
    return mols_with_attachs


def get_mols_with_inserts(target_point, mol0, addons):

    rdkit_mol0 = mol0.rdkit_mol
    poia0 = mol0.poia
    poih0 = mol0.poih
    poia_add0 = mol0.poia_add
    poih_add0 = mol0.poih_add
    hist0 = mol0.history

    mols_with_inserts = []

    target_exp_val = rdkit_mol0.GetAtomWithIdx(target_point).GetExplicitValence()

    for addon in addons:
        mh_index = 0
        target_free_at = addon['rdkit_mol'].GetAtomWithIdx(mh_index).GetTotalValence() - \
            addon['rdkit_mol'].GetAtomWithIdx(mh_index).GetExplicitValence()
        if target_exp_val <= target_free_at:
            new_rdkitmol = rdkit_mol0.__copy__()
            poia = poia0.copy()
            poih = poih0.copy()
            poia_add = poia_add0.copy()
            poih_add = poih_add0.copy()
            hist = list(hist0)

            em = Chem.EditableMol(new_rdkitmol)

            em.ReplaceAtom(target_point, addon['rdkit_mol'].GetAtomWithIdx(mh_index))

            num_atom1 = len(new_rdkitmol.GetAtoms())

            for atom in addon['rdkit_mol'].GetAtoms():
                if atom.GetIdx() != mh_index:
                    em.AddAtom(atom)

            for bond in addon['rdkit_mol'].GetBonds():
                if bond.GetBeginAtomIdx() == mh_index:
                    em.AddBond(target_point, bond.GetEndAtomIdx() + num_atom1 - 1,
                               bond.GetBondType())
                elif bond.GetEndAtomIdx() == mh_index:
                    em.AddBond(bond.GetBeginAtomIdx() + num_atom1 - 1, target_point,
                               bond.GetBondType())
                else:
                    em.AddBond(bond.GetBeginAtomIdx() + num_atom1 - 1,
                               bond.GetEndAtomIdx() + num_atom1 - 1,
                               bond.GetBondType())
            outmol = em.GetMol()
            if rdkit_mol0.GetAtomWithIdx(target_point).GetIsAromatic():
                outmol.GetAtomWithIdx(target_point).SetIsAromatic(True)
            Chem.SanitizeMol(outmol)

            poia_add = [i for i in poia_add.tolist() if i != target_point]
            poia_add += [num_atom1 + i for i in addon['poia'] if i != 0]
            if 0 in addon['poia']:
                poia_add.append(target_point)
            poia_add = np.asarray(poia_add)

            poih_add = [i for i in poih_add.tolist() if i != target_point]
            poih_add += [num_atom1 + i for i in addon['poih'] if i != 0]
            if 0 in addon['poih']:
                poih_add.append(target_point)
            poih_add = np.asarray(poih_add)

            poia = np.delete(poia, np.where(poia == target_point), None)
            poih = np.delete(poih, np.where(poia == target_point), None)
            hist.append('i-' + str(target_point) + '-' + addon['name'])
            mol_out = Molecule({'poia': poia, 'poih': poih, 'poia_add': poia_add,
                                'poih_add': poih_add, 'smiles': Chem.MolToSmiles(outmol),
                                'history': hist, 'rdkit_mol': outmol})

            mols_with_inserts.append(mol_out)
    return mols_with_inserts
