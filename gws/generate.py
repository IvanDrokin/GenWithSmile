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
    invh, inva = symmetric.get_symmetric(m0, poih, poia)
    molh, mola = generate_1(m0, invh, inva, adds)
    list_mols = []
    list_mols_smiles = []
    mol_tmp = molh + mola
    list_mols += mol_tmp
    list_mols_smiles += get_list_of_smiles(mol_tmp)
    curr_m = np.hstack([mola, molh])
    for i in range(1, num_iter):
        nl_m = []
        nl_s = []
        nl_f = []
        for j in range(len(curr_m)):
            poih = curr_m[j].poih
            poia = curr_m[j].poia
            # Check symmetric
            invh, inva = symmetric.get_symmetric(curr_m[j], poih, poia)
            # generate_1
            molh, mola = generate_1(curr_m[j], invh, inva, adds)
            # cut
            # cut
            if is_test == 0:
                l_tmp = molh + mola
                if len(l_tmp) > 0:
                    nl_m, nl_s, nl_f = gk.graph_kernel(nl_m, nl_s, nl_f, l_tmp, gk_param)
            else:
                nl_m, nl_s = unconcat(nl_m, nl_s, molh, mola)
        curr_m = nl_m
        list_mols += nl_m
        list_mols_smiles += nl_s
    for i in range(len(list_mols)):
        list_mols[i].poia = list_mols[i].poia_add
        list_mols[i].poih = list_mols[i].poih_add
        list_mols[i].poih_add = []
        list_mols[i].poia_add = []
    return list_mols, list_mols_smiles


def unconcat(nl_m, nl_s, molh, mola):
    mol = molh + mola
    n = np.shape(mol)[0]
    k = np.shape(nl_m)[0]
    mask = np.ones((n, 1))
    for i in range(n):
        for j in range(k):
            somorph = graphs_isomorph_mol(mol[i], nl_m[j])
            if somorph > 0:
                mask[i] = 0
                break
    indx = np.where(mask == 1)[0]
    #  mol_tmp = [mol[i] for i in indx]
    nl_m += [mol[i] for i in indx]
    nl_s += [mol[i].smiles for i in indx]
    #  nl_s += get_list_of_smiles(mol_tmp)
    return nl_m, nl_s


def get_list_of_smiles(mol_tmp):
    smiles = []
    for i in range(len(mol_tmp)):
        smiles.append(mol_tmp[i].smiles)
    return smiles


def graphs_isomorph_mol(mol1, mol2):
    g1 = mol1.g + mol1.hb
    g2 = mol2.g + mol2.hb
    a1 = mol1.atom
    a2 = mol2.atom
    return internal.graphs_isomorph_atom(g1, g2, a1, a2)


def generate_1(mol, invh, inva, adds_in):
    #  {'g': gr, 'gh': gr2, 'atom': atom, 'atom_pos': ap, 'hb': hb, 'sb': sb, 'charge': charge, 'poia': poia}
    gm0 = mol.g
    gmh0 = mol.gh
    ma0 = mol.atom
    ap0 = mol.atom_pos
    hb0 = mol.hb
    sb0 = mol.sb
    chaarge0 = mol.charge
    smiles0 = mol.smiles
    poia0 = mol.poia
    poih0 = mol.poih
    poia_add0 = mol.poia_add
    poih_add0 = mol.poih_add
    hist0 = mol.history

    adds = adds_in['attach']
    numadds = np.shape(adds)[0]
    numapoint = len(inva)
    numhpoint = len(invh)
    molh = []
    mola = []

    mh = np.sum(gmh0, 1)
    ma = np.sum(gm0, 1)

    for i in range(numhpoint):
        for j in range(numadds):
            mh_ad = adds[j]['bound']
            freevol = np.sum(adds[j]['gh'], 1)
            mh_add = mh_ad
            if mh[invh[i]] >= mh_ad and mh_ad <= freevol[0]:
                poia = poia0.copy()
                poih = poih0.copy()
                poia_add = poia_add0
                poih_add = poih_add0
                hist = list(hist0)

                gr = adds[j]['g'].copy()
                g = np.vstack([np.hstack([gm0, np.zeros((np.shape(gm0)[0], np.shape(gr)[1]))]),
                               np.hstack([np.zeros((np.shape(gr)[0], np.shape(gm0)[1])), gr])])
                g[invh[i], np.shape(gm0)[0]] = mh_add
                g[np.shape(gm0)[0], invh[i]] = mh_add

                grh = adds[j]['gh'].copy()
                grh[range(mh_add), range(mh_add)] = 0
                grh = np.delete(grh, range(mh_add), 1)  # grh[:, range(mh_add)] = []
                gh = np.vstack([np.hstack([gmh0, np.zeros((np.shape(gmh0)[0], np.shape(grh)[1]))]),
                                np.hstack([np.zeros((np.shape(grh)[0], np.shape(gmh0)[1])), grh])])
                ind = np.nonzero(gh[invh[i], :])[0][0]
                gh[invh[i], range(ind, ind + mh_add - 1)] = 0
                gh = np.delete(gh, np.where(np.sum(gh) == 0), 1)  # gh[:, np.nonzero(sum(gh) == 0)[0]] = []

                ar = adds[j]['atom'].copy()
                a = np.hstack([ma0, ar])

                chaarge = adds[j]['charge'].copy()
                chaarge = np.hstack([chaarge0, chaarge])

                hb_r = adds[j]['hb'].copy()
                hb = np.vstack([np.hstack([hb0, np.zeros((np.shape(hb0)[0], np.shape(hb_r)[1]))]),
                                np.hstack([np.zeros((np.shape(hb_r)[0], np.shape(hb0)[1])), hb_r])])

                sb_r = adds[j]['sb'].copy()
                sb = np.vstack([sb0, sb_r])

                apr = adds[j]['atom_pos'].copy()
                smilesr = adds[j]['smiles']
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

                sm = smiles0[0:(ap0[invh[i], 1] + 1)] + '(' + bound_sm + smilesr + ')' +\
                    smiles0[(1 + ap0[invh[i], 1]):len(smiles0)]
                #  sm = np.concatenate([smiles0[0:(ap0[invh[i], 1] + 1)], '(', smilesr, ')',
                #                    smiles0[range((1+ap0[invh[i], 1]), len(smiles0))]])

                s1 = ap0[range(invh[i] + 1), :]
                s2 = ap0[range(invh[i] + 1, np.shape(ap0)[0]), :]
                ls1 = len(smiles0[0:(ap0[invh[i], 1]+1)])
                lapr = len(smilesr)
                d = ap0[invh[i], 1] - ap0[invh[i], 0] + 1
                ap = np.vstack([s1, s2 + 2 + lapr + boind_off, apr + ls1 + d + boind_off])
                tmp_vec = range((np.shape(gmh0)[0]), (np.shape(gmh0)[0] + np.shape(gr)[0]))
                #  inva_next = np.hstack([poia, tmp_vec])

                poia_add = np.hstack([tmp_vec, poia_add])
                poih_add = np.hstack([tmp_vec, poih_add])

                index = np.argsort(ap[:, 0])
                ap = ap[index, :]
                a = a[index]
                sb = sb[index]
                g = g[index, :]
                g = g[:, index]
                hb = hb[index, :]
                hb = hb[:, index]
                gh = gh[index, :]

                poia = np.array([p for p in range(len(index)) for r in range(len(poia)) if poia[r] == index[p]])
                poih = np.array([p for p in range(len(index)) for r in range(len(poih)) if poih[r] == index[p]])
                poia_add = np.array([p for p in range(len(index)) for r in range(len(poia_add))
                                     if poia_add[r] == index[p]])
                poih_add = np.array([p for p in range(len(index)) for r in range(len(poih_add))
                                     if poih_add[r] == index[p]])

                hist.append('a-' + adds[j]['name'])
                ghs = np.sum(gh, 1)
                ghs = ghs.astype('int')
                gh = np.zeros(np.shape(gh))
                off = 0L
                for k in range(len(ghs)):
                    gh[k, range(off, (off + ghs[k]))] = 1
                    off = off + ghs[k]
                mol_out = Molecule({'g': g, 'gh': gh, 'atom': a, 'atom_pos': ap, 'hb': hb, 'sb': sb, 'charge': chaarge,
                           'poia': poia, 'poih': poih, 'poia_add': poia_add, 'poih_add': poih_add, 'smiles': sm,
                           'history': hist})
                molh.append(mol_out)

    adds = adds_in['insert']
    numadds = np.shape(adds)[0]
    for i in range(numapoint):
        for j in range(numadds):
            mh_ad = np.sum(adds[j]['gh'], 1)[0]  # + sum(adds[j]['g'], 1)[0]
            mh_ad = mh_ad.astype('int')
            if ma[inva[i]] <= mh_ad:
                poia = poia0.copy()
                poih = poih0.copy()
                poia_add = poia_add0
                poih_add = poih_add0
                hist = list(hist0)

                gr = adds[j]['g'].copy()
                gr1r = gr[0, 1:]
                gr1c = gr[1:, 0]
                gr = gr[1:, 1:]
                g = np.vstack([np.hstack([gm0, np.zeros((np.shape(gm0)[0], np.shape(gr)[1]))]),
                               np.hstack([np.zeros((np.shape(gr)[0], np.shape(gm0)[1])), gr])])
                g[inva[i], np.shape(gm0)[1]:] = gr1r
                g[np.shape(gm0)[0]:, inva[i]] = gr1c

                mh_add = ma[inva[i]]
                mh_add = mh_add.astype('int')
                grh = adds[j]['gh'].copy()
                grh[0, range(mh_add)] = 0
                grh = np.delete(grh, range(mh_ad), 1)  # grh[:, range(mh_ad)] = []
                grh = np.delete(grh, 0, 0)  # grh[0, :] = []
                gh = np.vstack([np.hstack([gmh0, np.zeros((np.shape(gmh0)[0], np.shape(grh)[1]))]),
                                np.hstack([np.zeros((np.shape(grh)[0], np.shape(gmh0)[1])), grh])])

                gh[inva[i], :] = 0
                gh = np.delete(gh, np.where(np.sum(gh, 0) == 0), 1)  # gh[:, np.where(sum(gh) == 0)] = []
                if mh_ad - mh_add > 0:
                    add_h_arr = np.zeros((np.shape(gh)[0], (mh_ad - mh_add)))
                    add_h_arr[inva[i], :] = 1
                    gh = np.hstack([gh, add_h_arr])

                ar = adds[j]['atom'].copy()
                a = np.hstack([ma0, ar[1:]])
                a[inva[i]] = ar[0]

                chaarge_r = adds[j]['charge'].copy()
                chaarge = np.hstack([chaarge0, chaarge_r[1:]])
                chaarge[inva[i]] = chaarge_r[0]

                hbr = adds[j]['hb'].copy()
                hb1r = hbr[0, 1:]
                hb1c = hbr[1:, 0]
                hbr = hbr[1:, 1:]
                hb = np.vstack([np.hstack([hb0, np.zeros((np.shape(hb0)[0], np.shape(hbr)[1]))]),
                                np.hstack([np.zeros((np.shape(hbr)[0], np.shape(hb0)[1])), hbr])])

                hb[inva[i], np.shape(hb0)[1]:] = hb1r
                hb[np.shape(hb0)[0]:, inva[i]] = hb1c

                sb_r = adds[j]['sb'].copy()
                if np.shape(sb_r[1:])[0] > 0:
                    sb = np.vstack([sb0, sb_r[1:]])
                else:
                    sb = sb0
                sb[inva[i]] = sb_r[0]

                if np.sum(hb0[inva[i], :] > 3) == 0:
                    apr = adds[j]['atom_pos'].copy()
                    smilesr = adds[j]['smiles']
                    sm = smiles0[0:ap0[inva[i], 0]] + smilesr[apr[0, 0]:(apr[0, 1] + 1)] + '(' + \
                        smilesr[(apr[0, 1] + 1):len(smilesr)] + ')' + smiles0[(1 + ap0[inva[i], 1]):len(smiles0)]
                    s1 = ap0[range(inva[i]), :].copy()
                    s2 = ap0[inva[i] + 1:, :].copy()
                    s2 -= (ap0[inva[i], 1] - ap0[inva[i], 0] + 1)
                    ls1 = len(smiles0[0:(ap0[inva[i], 0])])
                    lapr = len(smilesr)
                    apr[1:, :] = apr[1:, :] + 1
                    ap = np.vstack([s1, apr[0, :] + ls1, s2 + 2 + lapr, apr[1:, :] + ls1])
                else:
                    apr = adds[j]['atom_pos'].copy()
                    smilesr = adds[j]['smiles']
                    ind_a = np.nonzero(hb0[inva[i], :] > 3)[0]

                    if ma0[inva[i]].upper() == 'H':
                        sm0_tmp = smiles0[0:ap0[inva[i], 0]] + smiles0[(ap0[inva[i], 1] + 1):]
                        ap0_tmp = ap0.copy()
                        ind_att = max(ap0[inva[i], 1], ap0[ind_a, 1]) + 2
                        ap0_tmp[(inva[i] + 1):, :] = ap0_tmp[(inva[i] + 1):, :] - (
                            ap0[inva[i], 1] - ap0[inva[i], 0] + 1)
                        ind_att -= (ap0[inva[i], 1] - ap0[inva[i], 0] + 1)
                        sm = sm0_tmp[0:ind_att] + '(' + smilesr + ')' + smiles0[(ind_att+1):]
                        ind_att += 1
                        ap0_tmp[inva[i] + 1:, :] += 2 + len(smilesr)
                        ap0_tmp[inva[i], :] = apr[0, :] + ind_att
                        ap = np.vstack([ap0_tmp, (apr[1:, :] + ind_att)])
                    else:
                        sm = smiles0[0:(ap0[inva[i], 0] - 1)] + smilesr[apr[0, 0]:(apr[0, 1] + 1)] + '(' +\
                            smilesr[(apr[0, 1] + 1):] + ')' + smiles0[(1 + ap0[inva[i], 1]):]
                        s1 = ap0[range(inva[i] + 1), :].copy()
                        s2 = ap0[(1 + inva[i]):, :].copy()
                        s2 -= (ap0[inva[i], 1] - ap0[inva[i], 0] + 1)
                        ls1 = len(smiles0[0:(ap0[inva[i], 1])])
                        lapr = len(smilesr)
                        apr[1:, :] += 1
                        ap = np.vstack([s1, apr + ls1, s2 + 2 + lapr])
                # Check for empty () in smiles
                token = '\(\)'
                s1 = re.finditer(token, sm)
                for p in s1:
                    sm = sm[0:p.start()] + sm[p.end():]
                    for t in range(np.shape(ap)[0]):
                        if ap[t, 0] >= p.end():
                            ap[t, ] -= 2

                poi = poia
                np.delete(poi, np.where(poi == inva[i]))  # poi[poi == inva[i]] = []
                poia_add = np.hstack([poia_add, range((np.shape(gmh0)[0] + 1),
                                                      (np.shape(gmh0)[0] + np.shape(gr)[0] + 1))])
                poih_add = np.hstack([poih_add, range((np.shape(gmh0)[0] + 1),
                                                      (np.shape(gmh0)[0] + np.shape(gr)[0] + 1))])
                poia = np.delete(poia, np.where(poia == inva[i]))
                poih = np.delete(poih, np.where(poih == inva[i]))
                index = np.argsort(ap[:, 0])
                ap = ap[index, :]
                a = a[index]
                sb = sb[index]
                g = g[index, :]
                g = g[:, index]
                hb = hb[index, :]
                hb = hb[:, index]
                gh = gh[index, :]

                poia = np.array([p for p in range(len(index)) for r in range(len(poia)) if poia[r] == index[p]])
                poih = np.array([p for p in range(len(index)) for r in range(len(poih)) if poih[r] == index[p]])
                poia_add = np.array([p for p in range(len(index)) for r in range(len(poia_add))
                                     if poia_add[r] == index[p]])
                poih_add = np.array([p for p in range(len(index)) for r in range(len(poih_add))
                                     if poih_add[r] == index[p]])

                hist.append('i-' + adds[j]['name'])
                ghs = np.sum(gh, 1)
                ghs = ghs.astype('int')
                gh = np.zeros(np.shape(gh))
                off = 0L
                for k in range(len(ghs)):
                    gh[k, range(off, (off + ghs[k]))] = 1
                    off = off + ghs[k]
                mol_out = Molecule({'g': g, 'gh': gh, 'atom': a, 'atom_pos': ap, 'hb': hb, 'sb': sb, 'charge': chaarge,
                           'poia': poia, 'poih': poih, 'poia_add': poia_add, 'poih_add': poih_add, 'smiles': sm,
                           'history': hist})
                mola.append(mol_out)

    return molh, mola