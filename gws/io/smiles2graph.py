# coding=utf-8
import re
import numpy as np
from pyper import *


def smiles2graph(smiles_string, aromatic=0):
    # ==================================
    # Test input
    # aromatic = 1
    # smiles_string = 'c1ccncc1c2ccccc2'
    # ==================================
    # TODO to openBabel python
    rs = R()
    rs('a <- %s' % Str4R(smiles_string))
    rs('library("ChemmineR")')
    rs('smiset = as(a, "SMIset")')
    rs('sdfset = smiles2sdf(smiset)')
    rs('ab = atomblock(sdfset[[1]])')
    rs('bb = bondblock(sdfset[[1]])')
    rs('cb = bonds(sdfset[[1]]);')
    rs('hb = bonds(sdfset[[1]], type="addNH")')
    rs('at_names = row.names(ab);')
    rs('rings <- rings(sdfset[[1]], upper=7, type="all", arom=TRUE, inner=FALSE)')
    rs('r = rings[[1]]')
    rs('rt = sapply(rings[[2]],function(x){ifelse(x,1,0)})')
    bond = np.array(rs['bb'])
    # cb = rs['cb']
    num_h = rs['hb']
    an = rs['at_names']
    r = rs['r']
    rt = (rs['rt'])
    if isinstance(rt, int):
        rt = [rt]
    atom = rs['cb$atom']
    nbondcount = rs['cb$Nbondcount']
    nbondrule = rs['cb$Nbondrule']
    charge = rs['cb$charge']
    # end openBabel python
    num_vertex = np.amax(bond[:, [0, 1]])
    gr = np.zeros((num_vertex, num_vertex))
    hb = np.zeros((num_vertex, num_vertex))

    bond[:, [0, 1]] = bond[:, [0, 1]] - 1
    for i in range(bond.shape[0]):
        if bond[i, 3] > 0:
            if bond[i, 3] == 1:  # @
                gr[bond[i, 0], bond[i, 1]] = bond[i, 2]
                hb[bond[i, 0], bond[i, 1]] = 4
            else:
                # bond(i,4) == 6
                gr[bond[i, 0], bond[i, 1]] = bond[i, 2]
                hb[bond[i, 0], bond[i, 1]] = 8
        else:
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
    auniq = list(set(atom))
    ind = np.zeros((len(atom)*2), dtype=int)
    cnt = 0

    aind = np.zeros((2 * len(atom)), dtype='object')
    for i in range(len(auniq)):
        k = [i for i in range(len(smiles_string.upper())) if smiles_string.upper().startswith(auniq[i].upper(), i)]
        ind[cnt:(cnt + len(k))] = k
        aind[cnt:(cnt + len(k))] = auniq[i]
        cnt += len(k)
    ind = ind[:cnt]
    aind = aind[:cnt]
    ix = np.argsort(ind, 0)
    ind = ind[ix]
    aind = aind[ix]

    cnt = 0
    while True:
        # ind_tmp = np.array((ind == ind[cnt]).nonzero())
        # ind_tmp = np.array((ind == ind[cnt]).ravel().nonzero())
        ind_tmp = np.where(ind == ind[cnt])[0]
        if len(ind_tmp) > 1:
            l2 = np.zeros((len(ind_tmp)), dtype=int)
            for j in range(len(ind_tmp)):
                l2[j] = len(aind[ind_tmp[j]])
            in2 = np.argmax(l2)
            ind_tmp = np.delete(ind_tmp, in2)  # ind_tmp[in2] = []
            ind = np.delete(ind, ind_tmp)  # ind[ind_tmp] = []
            aind = np.delete(aind, ind_tmp)  # aind[ind_tmp] = []
            cnt = 0
        else:
            cnt += 1
            if cnt >= len(ind):
                break

    for i in range(len(aind)):
        if (ind[i] == 0) or ((ind[i] + len(aind[i])) == len(aind)):
            ap[i, 0] = ind[i]
            ap[i, 1] = ind[i] + len(aind[i]) - 1
        else:
            f = ind[i]
            l = ind[i] + len(aind[i]) - 1
            # Isotope check
            token = '\[[0-9]{1,}' + str(aind[i]) + '\]'
            s1 = re.finditer(token, smiles_string)
            for m in s1:
                if m.start(0) < f and l < m.end(0):
                    l = m.end(0)
                    f = m.start(0)
                    break
            ap[i, 0] = f
            ap[i, 1] = l
            # ion check
            token = '\[' + str(aind[i]) + '\+{1,}[0-9]{0,}\]'
            s1 = re.finditer(token, smiles_string)
            for m in s1:
                if m.start(0) < f and l < m.end(0):
                    l = m.end(0)
                    f = m.start(0)
                    break
            ap[i, 0] = f
            ap[i, 1] = l
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
