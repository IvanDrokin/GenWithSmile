# coding=utf-8
import re
import numpy as np

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
        list_mols[i].poih_add = []
        list_mols[i].poia_add = []
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
    g1 = mol1.g
    g2 = mol2.g
    a1 = mol1.atom
    a2 = mol2.atom
    return internal.graphs_isomorph_atom(g1, g2, a1, a2)


def generate_1(mol, invh, inva, adds_in):

    generated_mols = []
    for i in invh:
        generated_mols += get_mols_with_attachs(i, mol, adds_in['attach'])

    for i in inva:
        generated_mols += get_mols_with_inserts(i, mol, adds_in['insert'])

    return generated_mols


def get_mols_with_attachs(target_point, mol0, addons):

    gm0 = mol0.g
    gmh0 = mol0.gh
    ma0 = mol0.atom
    ap0 = mol0.atom_pos
    chiral_tags0 = mol0.chiral_tags
    chaarge0 = mol0.charge
    smiles0 = mol0.smiles
    poia0 = mol0.poia
    poih0 = mol0.poih
    poia_add0 = mol0.poia_add
    poih_add0 = mol0.poih_add
    hist0 = mol0.history

    mols_with_attachs = []

    mh = np.sum(gmh0, 1)
    for attach in addons:
        mh_ad = attach['bound']
        freevol = np.sum(attach['gh'], 1)
        mh_add = mh_ad
        if mh[target_point] >= mh_ad and mh_ad <= freevol[0]:
            poia = poia0.copy()
            poih = poih0.copy()
            poia_add = poia_add0
            poih_add = poih_add0
            hist = list(hist0)

            gr = attach['g'].copy()
            g = np.vstack([np.hstack([gm0, np.zeros((np.shape(gm0)[0], np.shape(gr)[1]))]),
                           np.hstack([np.zeros((np.shape(gr)[0], np.shape(gm0)[1])), gr])])
            g[target_point, np.shape(gm0)[0]] = mh_add
            g[np.shape(gm0)[0], target_point] = mh_add

            grh = attach['gh'].copy()
            grh[range(mh_add), range(mh_add)] = 0
            grh = np.delete(grh, range(mh_add), 1)  # grh[:, range(mh_add)] = []
            gh = np.vstack([np.hstack([gmh0, np.zeros((np.shape(gmh0)[0],
                                                       np.shape(grh)[1]), dtype=int)]),
                            np.hstack([np.zeros((np.shape(grh)[0],
                                                 np.shape(gmh0)[1]), dtype=int), grh])])
            ind = np.nonzero(gh[target_point, :])[0][0]
            gh[target_point, range(ind, ind + mh_add)] = 0
            gh = np.delete(gh, np.where(np.sum(gh) == 0), 1)

            ar = attach['atom'].copy()
            a = np.hstack([ma0, ar])

            chaarge = attach['charge'].copy()
            chaarge = np.hstack([chaarge0, chaarge])

            chiral_tags_r = attach['chiral_tags'].copy()
            chiral_tags = np.hstack([chiral_tags0, chiral_tags_r])

            apr = attach['atom_pos'].copy()
            smilesr = attach['smiles']
            # sm = smiles0[range(ap0[invh[i], 1])] + '(' + smilesr ')' +
            # smiles0[range((1+ap0[invh[i], 1]), len(smiles0))]
            bound_sm = ''
            boind_off = 0
            if mh_ad == 2:
                boind_off = 1
                bound_sm = '='
            if mh_ad == 3:
                boind_off = 1
                bound_sm = '#'

            sm = smiles0[0:(ap0[target_point, 1] + 1)] + '(' + bound_sm + smilesr + ')' +\
                smiles0[(1 + ap0[target_point, 1]):len(smiles0)]
            #  sm = np.concatenate([smiles0[0:(ap0[invh[i], 1] + 1)], '(', smilesr, ')',
            #                    smiles0[range((1+ap0[invh[i], 1]), len(smiles0))]])

            s1 = ap0[range(target_point + 1), :]
            s2 = ap0[range(target_point + 1, np.shape(ap0)[0]), :]
            ls1 = len(smiles0[0:(ap0[target_point, 1]+1)])
            lapr = len(smilesr)
            d = ap0[target_point, 1] - ap0[target_point, 0] + 1
            ap = np.vstack([s1, s2 + 2 + lapr + boind_off, apr + ls1 + d + boind_off])
            #  inva_next = np.hstack([poia, tmp_vec])

            tmp_poia = attach['poia'].copy() + ap0.shape[0]
            tmp_poih = attach['poih'].copy() + ap0.shape[0]

            poia_add = np.hstack([tmp_poia, poia_add])
            poih_add = np.hstack([tmp_poih, poih_add])

            index = np.argsort(ap[:, 0])
            ap = ap[index, :]
            a = a[index]
            g = g[index, :]
            g = g[:, index]
            gh = gh[index, :]

            poia = np.array([p for p in range(len(index))
                             for r in range(len(poia)) if poia[r] == index[p]])
            poih = np.array([p for p in range(len(index))
                             for r in range(len(poih)) if poih[r] == index[p]])
            poia_add = np.array([p for p in range(len(index))
                                 for r in range(len(poia_add))
                                 if poia_add[r] == index[p]])
            poih_add = np.array([p for p in range(len(index))
                                 for r in range(len(poih_add))
                                 if poih_add[r] == index[p]])

            hist.append('a-' + str(target_point) + '-' + attach['name'])
            ghs = np.sum(gh, 1)
            gh = np.zeros(np.shape(gh), dtype=int)
            off = 0L
            for k in range(len(ghs)):
                gh[k, range(off, (off + ghs[k]))] = 1
                off = off + ghs[k]
            mol_out = Molecule({'g': g, 'gh': gh, 'atom': a, 'atom_pos': ap,
                                'chiral_tags': chiral_tags, 'charge': chaarge, 'poia': poia,
                                'poih': poih, 'poia_add': poia_add, 'poih_add': poih_add,
                                'smiles': sm, 'history': hist})
            mols_with_attachs.append(mol_out)
    return mols_with_attachs


def get_mols_with_inserts(target_point, mol0, addons):

    gm0 = mol0.g
    gmh0 = mol0.gh
    ma0 = mol0.atom
    ap0 = mol0.atom_pos
    chiral_tags0 = mol0.chiral_tags
    chaarge0 = mol0.charge
    smiles0 = mol0.smiles
    poia0 = mol0.poia
    poih0 = mol0.poih
    poia_add0 = mol0.poia_add
    poih_add0 = mol0.poih_add
    hist0 = mol0.history

    ma = np.sum(gm0, 1)

    mols_with_inserts = []

    for addon in addons:
        mh_ad = np.sum(addon['gh'], 1)[0]  # + sum(adds[j]['g'], 1)[0]
        mh_ad = mh_ad.astype('int')
        if ma[target_point] <= mh_ad:
            poia = poia0.copy()
            poih = poih0.copy()
            poia_add = poia_add0
            poih_add = poih_add0
            hist = list(hist0)

            gr = addon['g'].copy()
            gr1r = gr[0, 1:]
            gr1c = gr[1:, 0]
            gr = gr[1:, 1:]
            g = np.vstack([np.hstack([gm0, np.zeros((np.shape(gm0)[0], np.shape(gr)[1]))]),
                           np.hstack([np.zeros((np.shape(gr)[0], np.shape(gm0)[1])), gr])])
            g[target_point, np.shape(gm0)[1]:] = gr1r
            g[np.shape(gm0)[0]:, target_point] = gr1c

            mh_add = ma[target_point]
            mh_add = mh_add.astype('int')
            grh = addon['gh'].copy()
            grh[0, range(mh_add)] = 0
            grh = np.delete(grh, range(mh_ad), 1)
            grh = np.delete(grh, 0, 0)
            gh = np.vstack([np.hstack([gmh0, np.zeros((np.shape(gmh0)[0],
                                                       np.shape(grh)[1]), dtype=int)]),
                            np.hstack([np.zeros((np.shape(grh)[0],
                                                 np.shape(gmh0)[1]), dtype=int), grh])])

            gh[target_point, :] = 0
            gh = np.delete(gh, np.where(np.sum(gh, 0) == 0), 1)
            if mh_ad - mh_add > 0:
                add_h_arr = np.zeros((np.shape(gh)[0], (mh_ad - mh_add)), dtype=int)
                add_h_arr[target_point, :] = 1
                gh = np.hstack([gh, add_h_arr])

            ar = addon['atom'].copy()
            a = np.hstack([ma0, ar[1:]])
            a[target_point] = ar[0]

            chiral_tags_r = addon['chiral_tags'].copy()
            chiral_tags = np.hstack([chiral_tags0, chiral_tags_r[1:]])
            chiral_tags[target_point] = chiral_tags_r[0]

            chaarge_r = addon['charge'].copy()
            chaarge = np.hstack([chaarge0, chaarge_r[1:]])
            chaarge[target_point] = chaarge_r[0]

            apr = addon['atom_pos'].copy()
            smilesr = addon['smiles']
            sm = smiles0[0:ap0[target_point, 0]] + smilesr[apr[0, 0]:(apr[0, 1] + 1)] + '(' + \
                smilesr[(apr[0, 1] + 1):len(smilesr)] + ')' + \
                smiles0[(1 + ap0[target_point, 1]):len(smiles0)]
            s1 = ap0[range(target_point), :].copy()
            s2 = ap0[target_point + 1:, :].copy()
            s2 -= (ap0[target_point, 1] - ap0[target_point, 0] + 1)
            ls1 = len(smiles0[0:(ap0[target_point, 0])])
            lapr = len(smilesr)
            apr[1:, :] = apr[1:, :] + 1
            ap = np.vstack([s1, apr[0, :] + ls1, s2 + 2 + lapr, apr[1:, :] + ls1])

            # Check for empty () in smiles
            token = '\(\)'
            s1 = re.finditer(token, sm)
            for p in s1:
                sm = sm[0:p.start()] + sm[p.end():]
                for t in range(np.shape(ap)[0]):
                    if ap[t, 0] >= p.end():
                        ap[t, ] -= 2

            poi = poia
            np.delete(poi, np.where(poi == target_point))  # poi[poi == inva[i]] = []
            # TODO исключение инсертнутого атома из списка, если он в {} или
            tmp_poia = addon['poia'][np.where(addon['poia'] > 0)].copy()  # Исключаем "воткнутый" атом
            tmp_poih = addon['poih'].copy()
            tmp_poih[np.where(addon['poia'] == 0)] = target_point
            tmp_poih[np.where(addon['poia'] > 0)] += ap0.shape[0]
            tmp_poia += ap0.shape[0]
            poia_add = np.hstack([poia_add, tmp_poia])
            poih_add = np.hstack([poih_add, tmp_poih])
            poia = np.delete(poia, np.where(poia == target_point))
            poih = np.delete(poih, np.where(poih == target_point))
            index = np.argsort(ap[:, 0])
            ap = ap[index, :]
            a = a[index]
            g = g[index, :]
            g = g[:, index]
            gh = gh[index, :]

            poia = np.array([p for p in range(len(index))
                             for r in range(len(poia)) if poia[r] == index[p]])
            poih = np.array([p for p in range(len(index))
                             for r in range(len(poih)) if poih[r] == index[p]])
            poia_add = np.array([p for p in range(len(index))
                                 for r in range(len(poia_add))
                                 if poia_add[r] == index[p]])
            poih_add = np.array([p for p in range(len(index))
                                 for r in range(len(poih_add))
                                 if poih_add[r] == index[p]])

            hist.append('i-' + str(target_point) + '-' + addon['name'])
            ghs = np.sum(gh, 1)
            gh = np.zeros(np.shape(gh), dtype=int)
            off = 0L
            for k in range(len(ghs)):
                gh[k, range(off, (off + ghs[k]))] = 1
                off = off + ghs[k]
            mol_out = Molecule({'g': g, 'gh': gh, 'atom': a, 'atom_pos': ap,
                                'chiral_tags': chiral_tags, 'charge': chaarge, 'poia': poia,
                                'poih': poih, 'poia_add': poia_add, 'poih_add': poih_add,
                                'smiles': sm, 'history': hist})
            mols_with_inserts.append(mol_out)
    return mols_with_inserts