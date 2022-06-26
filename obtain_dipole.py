import sys
from METHOD_YOU_WILL_HAVE_TO_IMPLEMENT import read_dipole

#Obtain dipoles operator matrices in the atomic orbital basis.
#Requires the user of this program to provide read_dipole
def xyzmatrix(path):
    x = read_dipole(1, 0, 0)
    y = read_dipole(0, 1, 0)
    z = read_dipole(0, 0, 1)
    qc_pipe.request(0)
    with open(path + 'dipole.0', 'wb') as f:
        x.tofile(f)
        y.tofile(f)
        z.tofile(f)


if __name__ == '__main__':
    store_dipole_matrix(path)
