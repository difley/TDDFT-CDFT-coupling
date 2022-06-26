#!/usr/bin/env python2.5

from numpy import asarray, transpose, dot, trace
from numpy.linalg import det, inv
from coupling import get_tddft_amps, get_ol, get_cdft_v, get_mo, epsilon_states, one_body


def epsilon_states(path1, exn, nocc, epsilon):
    ol,dim = get_ol(path1)
    nvirt = dim - nocc
    mo = get_mo(path1, dim)
    moa = mo[:dim]
    mob = mo[dim:]
    rpa = 1 #set to 1 if RPA used, else assume TDA used
    ampsa, ampsb = get_tddft_amps(path1, nocc, rpa)
    for i in range(nocc):
        for a in range(nvirt):
            moa[i] += ampsa[exn, nvirt*i + a]*moa[nocc + a]*epsilon
            mob[i] += ampsb[exn, nvirt*i + a]*mob[nocc + a]*epsilon
    return moa[:nocc], mob[:nocc]  #moa[:nocc, :dim], mob[:nocc, :dim]

#compute <mo1|mo2>
def zero_body(ol, mo1, mo2):
    tot = 0.
    for i in range(len(mo1[0])):
        for j in range(len(mo2[0])):
            deta = det(dot(dot(mo1[1][i][0], ol), transpose(mo2[1][j][0])))
            detb = det(dot(dot(mo1[1][i][1], ol), transpose(mo2[1][j][1])))
            tot += deta*detb*mo1[0][i]*mo2[0][j]
    return tot

#compute <mo1|V|mo2>
def one_body(ol, mo1, mo2, v):
    tot = 0.
    for i in range(len(mo1[0])):
        for j in range(len(mo2[0])):
            sa = dot(dot(mo1[1][i][0], ol), transpose(mo2[1][j][0]))
            sb = dot(dot(mo1[1][i][1], ol), transpose(mo2[1][j][1]))
            vamo = dot(dot(mo1[1][i][0], v[0]), transpose(mo2[1][j][0]))
            vbmo = dot(dot(mo1[1][i][1], v[1]), transpose(mo2[1][j][1]))
            tot += det(sa)*det(sb)*trace(dot(inv(sa), vamo) + dot(inv(sb), vbmo))*mo1[0][i]*mo2[0][j]
    return tot

#compute <td|td>
def td_td(pathtd, nocc, rpa, exn, epsilon):
    ol, dim = get_ol(pathtd)
    mopa, mopb = epsilon_states(pathtd, exn, nocc, epsilon)
    mona, monb = epsilon_states(pathtd, exn, nocc, -epsilon)
    return zero_body(ol,
                     [asarray([0.5/epsilon, -0.5/epsilon]), [[mopa,mopb], [mona, monb]]],
                     [asarray([0.5/epsilon, -0.5/epsilon]), [[mopa,mopb], [mona, monb]]])

#compute <ct|ct>
def ct_ct(pathct, nocc, epsilon):
    ol, dim = get_ol(pathct)
    mocta = get_mo(pathct, dim)[:nocc]
    moctb = get_mo(pathct, dim)[dim:dim + nocc]
    return zero_body(ol,
                     [asarray([1.]), [[mocta, moctb]]],
                     [asarray([1.]), [[mocta, moctb]]])

#compute <ct|td>
def ct_td(pathct, pathtd, nocc, rpa, exn, epsilon):
    ol, dim = get_ol(pathct)
    mocta = get_mo(pathct, dim)[:nocc]
    moctb = get_mo(pathct, dim)[dim:dim + nocc]
    mopa, mopb = epsilon_states(pathtd, exn, nocc, epsilon)
    mona, monb = epsilon_states(pathtd, exn, nocc, -epsilon)
    return zero_body(ol,
                     [asarray([1.]), [[mocta,moctb]]],
                     [asarray([0.5/epsilon, -0.5/epsilon]), [[mopa, mopb], [mona, monb]]])

#compute <ct|V|ct>
def ct_v_ct(pathct, nocc, mult):
    ol,dim = get_ol(pathct)
    va, vb = get_cdft_v(pathct, dim, mult)
    moct = get_mo(pathct, dim)
    mocta = moct[:nocc]
    moctb = moct[dim:dim + nocc]
    return one_body(ol,
                    [asarray([1.]), [[mocta,moctb]]],
                    [asarray([1.]), [[mocta,moctb]]],
                    [va,vb])

#compute <ct|V|td>
def ct_v_td(pathct, pathtd, nocc, rpa, mult, exn, epsilon):
    ol, dim = get_ol(pathct)
    va, vb = get_cdft_v(pathct, dim, mult)
    moct = get_mo(pathct, dim)
    mocta = moct[:nocc]
    moctb = moct[dim:dim + nocc]
    mopa, mopb = epsilon_states(pathtd, exn, nocc, epsilon)
    mona, monb = epsilon_states(pathtd, exn, nocc, -epsilon)
    return one_body(ol,
                    [asarray([1.]), [[mocta, moctb]]],
                    [asarray([0.5/epsilon, -0.5/epsilon]), [[mopa, mopb], [mona, monb]]],
                    [va,vb])

#compute <td|V|td>
def td_v_td(pathct, pathtd, nocc, rpa, mult, exn, epsilon):
    ol,dim = get_ol(pathct)
    va, vb = get_cdft_v(pathct, dim, mult)
    moct = get_mo(pathct, dim)
    mocta = moct[:nocc]
    moctb = moct[dim:dim + nocc]
    mopa, mopb = epsilon_states(pathtd, exn, nocc, epsilon)
    mona, monb = epsilon_states(pathtd, exn, nocc, -epsilon)
    return one_body(ol,
                    [asarray([0.5/epsilon, -0.5/epsilon]), [[mopa, mopb], [mona, monb]]],
                    [asarray([0.5/epsilon, -0.5/epsilon]), [[mopa, mopb], [mona, monb]]],
                    [va,vb])
