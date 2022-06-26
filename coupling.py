#!/usr/bin/env python2.5
#Obtain overlap between two determinants
#computed using QChem.
#QChem job must be run with -save to retain needed files.

from numpy import asarray, fromfile, transpose, reshape, multiply as mm, shape, dot, trace, zeros
from math import sqrt
from numpy.linalg import det, inv

#nocc = 1  #number of occupied orbitals
#path = "/path/to/qchem/QCSCRATCH/subfolder/"


#Compute transition dipole moment for TDDFT excited states.  Calculation can be
#restricted or unrestricted and full TDDFT (rpa=1) or TD approximation (rpa=0)
#Dipoles in the AO basis must have been obtained previously using xyzmatrix()
#and stored in path/dipole.0.
def tmom(path, nocc, rpa):
    ol, dim = get_ol(path)
    nvirt = dim - nocc
    nvo = nocc*nvirt
    a = fromfile(path + 'dipole.0')
    x = reshape(a[:dim*dim], (dim, dim))
    y = reshape(a[dim*dim:2*dim*dim], (dim, dim))
    z = reshape(a[2 * dim* dim:], (dim, dim))
    rpa = True
    ampsa, ampsb = get_tddft_amps(path, nocc, rpa)
    mo = get_mo(path, dim)
    xmo = dot(dot(mo, x), transpose(mo))
    xmoa = reshape(xmo[:nocc, nocc:dim], (nvo))
    xmob = reshape(xmo[:nocc, dim + nocc:2*dim], (nvo))

    ymo = dot(dot(mo, y), transpose(mo))
    ymoa = reshape(ymo[:nocc, nocc:dim], (nvo))
    ymob = reshape(ymo[:nocc, dim + nocc:2*dim], (nvo))
    zmo = dot(dot(mo, z), transpose(mo))
    zmoa = reshape(zmo[:nocc, nocc:dim], (nvo))
    zmob = reshape(zmo[:nocc, dim + nocc:2*dim], (nvo))

    return transpose(asarray([dot(xmoa, transpose(ampsa)) +
                              dot(xmob,transpose(ampsb)), dot(ymoa,transpose(ampsa)) +
                              dot(ymob,transpose(ampsb)), dot(zmoa,transpose(ampsa)) +
                              dot(zmob,transpose(ampsb))]))


#Return vector of amplitudes corresponding to excited states in
#the following order:  alpha 0, beta 0, alpha 1, beta 1, alpha 2, beta 2, ...
#Thus, if n excited states have been computed and there are nocc occupied
#orbitals and nvirt virtual orbitals, a matrix with dimensions
#(n)X(nocc*nvirt).  E.g., amps[2] and amps[3] correspond to alpha 1 and
#beta 1 excited state amplitudes, respectively.
#In each vector, the "faster" index (inner loop index) is the virtual state.
#So we have o1v1,o1v2,o1v3,...,o2v1,o2v2,o2v3,...
def get_tddft_amps(path, nocc, rpa):
    ol,dim = get_ol(path)
    nvirt = dim - nocc
    nvo = nocc * nvirt

    if not rpa:
        amps = fromfile(path + "73.0", "d")
        amps = reshape(amps, (len(amps)/nvo, nvo))
    else:
        rpax = fromfile(path + "82.0", "d")
        rpax = reshape(rpax, (len(rpax)/nvo, nvo))

        rpay = fromfile(path + "83.0","d")
        rpay = reshape(rpay, (len(rpay)/nvo, nvo))

        amps = rpax + rpay
    return amps[0::2], amps[1::2]


def get_ol(path):
    ol = fromfile(path + "320.0", "d")
    dim = int(sqrt(len(ol)))
    ol = reshape(ol, (len(ol)/dim, dim))
    return ol, dim


def get_mo(path, dim):
    mo = fromfile(path + "53.0", "d")[:2*dim*dim]
    mo = reshape(mo, (len(mo)/dim, dim))
    return mo


#Returns cdft potential in ao basis
def get_cdft_v(path, dim, mult):
    tmp = fromfile(path + "605.0", "d")
    va = zeros(dim*dim, "d")
    vb = zeros(dim*dim, "d")
    for i in range(len(mult)): #allow for multiple constraints
        va += mult[i]*tmp[2*i*dim*dim + 1:(2*i + 1)*dim*dim + 1]
        vb += mult[i]*tmp[(2*i + 1)*dim*dim + 1:(2*i + 2)*dim*dim + 1]
    va = reshape(va, (dim, dim))
    vb = reshape(vb, (dim, dim))
    return va, vb


#Computes transition dipole moment and should return same numbers as in output file
def tmom_test(path, nocc, rpa=1):
    tmom(path, nocc, rpa)


#Computes mo*S*mo and should return identity
def ol_test(path):
    ol, dim = get_ol(path)
    mo = get_mo(path, dim)
    return dot(dot(mo, ol), transpose(mo))


#Return zero-body matrix element between two constrained states
def get_mo_olap(path1, path2, nocc):
    ol, dim = get_ol(path1)
    mo1 = get_mo(path1, dim)
    mo2 = get_mo(path2, dim)
    s1 = dot(dot(mo1, ol), transpose(mo2))  #overlap matrix
    #print s1
    sa = s1[:nocc, :nocc]
    va = s1[:nocc, nocc:dim]
    sb = s1[dim:dim + nocc, dim:dim + nocc]
    vb = s1[dim:dim + nocc, dim + nocc:2*dim]
    #print s1
    return sa, va, sb, vb


#Return <constrained state|tddft excited state> matrix element
#exn is the index of the excited state to use
#e.g. exn=0 is the lowest excited state
def get_tddft_zero_body(path1, path2, nocc, rpa, exn):
    sa, va, sb, vb = get_mo_olap(path1, path2, nocc)
    virt = va.shape[1]
    ampsa, ampsb = get_tddft_amps(path2, nocc, rpa) #tddft states are in path2

    result1 = 0.
    for i in range(nocc):
        for j in range(virt):
            temp = sa.copy()
            temp[:, i] = va[:, j]
            result1 += det(temp)*ampsa[exn, virt*i + j]
    result1 = result1*det(sb)

    result2 = 0.
    for i in range(nocc):
        for j in range(virt):
            temp = sb.copy()
            temp[:, i] = vb[:, j]
            result2 += det(temp)*ampsb[exn, virt*i + j]
    result2 = result2*det(sa)
    return result1+result2


def get_tddft_one_body(path1, path2, mult, nocc, rpa, exn):
    ol, dim = get_ol(path1)
    va, vb = get_cdft_v(path1, dim, mult)
    mo1 = get_mo(path1, dim)
    mo2 = get_mo(path2, dim)
    mo1a = mo1[:dim, :]
    mo1b = mo1[dim:, :]
    mo2a = mo2[:dim, :]
    mo2b = mo2[dim:, :]
    vamo = dot(dot(mo1a, va), transpose(mo2a))
    vbmo = dot(dot(mo1b, vb), transpose(mo2b))
    svbmo = vbmo[:nocc, :nocc]
    svamo = vamo[:nocc,:nocc]
    sa, virta, sb, virtb = get_mo_olap(path1, path2, nocc)
    virt = virta.shape[1]
    ampsa, ampsb = get_tddft_amps(path2, nocc, rpa) #tddft states are in path2

    result1 = 0.
    sbinv = inv(sb)
    db = det(sb)
    for i in range(nocc):
        for j in range(virt):
            temp = sa.copy()
            temp[:, i] = virta[:, j]
            tvamo = svamo.copy()
            tvamo[:,i] = vamo[:nocc,nocc+j]

            sainv = inv(temp)
            d = det(temp)*db
            result1 += d*trace(dot(sainv, tvamo) + dot(sbinv, svbmo))*ampsa[exn, virt*i + j]

    result2 = 0.

    sainv = inv(sa)
    da = det(sa)
    for i in range(nocc):
        for j in range(virt):
            temp = sb.copy()
            temp[:, i] = virtb[:, j]
            tvbmo = svbmo.copy()
            tvbmo[:, i] = vbmo[:nocc, nocc + j]
            sbinv = inv(temp)
            d = det(temp)*da
            result2 += trace(dot(d*sainv, svamo) + dot(d*sbinv, tvbmo))*ampsb[exn, virt*i + j]

    return result1 + result2


def gobt(outfile):
    ncons = 1 #number of constraints
    dim = 52 #number of basis functions
    pathtd = "/path/to/qchem/QCSCRATCH/tddft/"
    pathct = "/path/to/qchem/QCSCRATCH/cdft/"

    for i in range(10):
        v12 = get_twodet_zero_body(pathct, pathtd, 16, 1, i, 0.001)
        s12 = get_twodet_one_body(pathct, pathtd, 16, 1, 1, i, 0.001)
        v12 = get_tddft_one_body(pathct, pathtd, ncons, nocc, 1, i)
        s12 = get_tddft_zero_body(pathct, pathtd, nocc, 1, i)
        en = fromfile(pathct + "605.0", "d")[0]
        lam = fromfile(pathct + "605.0", "d")[2*dim*dim*ncons + 1 + ncons]
        nc = fromfile(pathct + "605.0", "d")[2*dim*dim*ncons + 1]
        with open(outfile, "a") as f:
            f.write("%f %f %f %f %f %f %i\n" % (c12, v12, s12, en, lam, nc, i))


#Should return the overlap between determinant states 1 and 2
def get_zero_body(path1, path2, nocc):
    a = []
    sa, va, sb, vb = get_mo_olap(path1, path1, nocc)
    a.append(det(sa)*det(sb))
    sa, va, sb, vb = get_mo_olap(path1, path2, nocc)
    a.append(det(sa)*det(sb))
    sa, va, sb, vb = get_mo_olap(path2, path1, nocc)
    a.append(det(sa)*det(sb))
    sa, va, sb, vb = get_mo_olap(path2, path2, nocc)
    a.append(det(sa)*det(sb))
    return reshape(asarray(a), (2, 2))


#Returns one-body matrix element
def get_one_body(path1, path2, nocc, mult):
    ol, dim = get_ol(path1)
    va, vb = get_cdft_v(path1, dim, mult)
    mo1 = get_mo(path1, dim)
    mo2 = get_mo(path2, dim)
    mo1a = mo1[:dim, :]
    mo1b = mo1[dim:, :]
    mo2a = mo2[:dim, :]
    mo2b = mo2[dim:, :]
    vamo = dot(dot(mo1a, va), transpose(mo2a))
    vbmo = dot(dot(mo1b, vb), transpose(mo2b))
    sa, virta, sb, virtb = get_mo_olap(path1, path2, nocc)
    sainv = inv(sa)
    sbinv = inv(sb)
    d = det(sa)*det(sb)
    sainv = d*sainv
    sbinv = d*sbinv
    return trace(dot(sainv, vamo[:nocc, :nocc])
               + dot(sbinv, vbmo[:nocc, :nocc])) #sum over occupied orbitals


def get_one_body_test(path1, path2, nocc, mult):
    print("<1|v|1> = %1.8f" % (get_one_body(path1, path1, nocc, mult)))
    print("<1|v|2> = %1.8f" % (get_one_body(path1, path2, nocc, mult)))
    print("<2|v|1> = %1.8f" % (get_one_body(path2, path1, nocc, mult)))
    print("<2|v|2> = %1.8f" % (get_one_body(path2, path2, nocc, mult)))

