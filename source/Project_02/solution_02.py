import numpy as np
from molmass import ELEMENTS
from scipy import constants
from scipy.constants import physical_constants
np.set_printoptions(precision=7, linewidth=120, suppress=True)

E_h = physical_constants["Hartree energy"][0]
a_0 = physical_constants["Bohr radius"][0]
amu = physical_constants["atomic mass constant"][0]
c_0 = physical_constants["speed of light in vacuum"][0]
centi = constants.centi


class Molecule:

    def __init__(self):
        self.atom_charges = NotImplemented  # type: np.ndarray
        self.atom_coords = NotImplemented  # type: np.ndarray
        self.natm = NotImplemented  # type: int
        self.hess = NotImplemented  # type: np.ndarray

    def construct_from_dat_file(self, file_path: str):
        # Same to Project 01
        with open(file_path, "r") as f:
            dat = np.array([line.split() for line in f.readlines()][1:])
            self.atom_charges = np.array(dat[:, 0], dtype=float).astype(int)
            self.atom_coords = np.array(dat[:, 1:4], dtype=float)
            self.natm = self.atom_charges.shape[0]

    def obtain_hessian(self, file_path: str):
        # Input: Read Hessian file from `file_path`
        # Attribute modification: Obtain raw Hessian to `self.hess`
        with open(file_path, "r") as f:
            self.hess = np.array([line.split() for line in f.readlines()][1:], dtype=float).reshape((self.natm * 3, self.natm * 3))

    def mass_weighted_hess(self) -> np.ndarray or list:
        # Output: Mass-weighted Hessian matrix (in unit Eh/(amu*a0^2))
        atom_weights = np.array([ELEMENTS[c].mass for c in self.atom_charges])
        freq_weights = np.sqrt(np.repeat(atom_weights, 3))
        return self.hess / freq_weights[:, None] / freq_weights[None, :]

    def eig_mass_weight_hess(self) -> np.ndarray or list:
        # Output: Eigenvalue of mass-weighted Hessian matrix (in unit Eh/(amu*a0^2))
        return np.linalg.eigvalsh(self.mass_weighted_hess())

    def harmonic_vib_freq(self) -> np.ndarray or list:
        # Output: Harmonic vibrational frequencies (in unit cm^-1)
        coef = np.sqrt(E_h / amu / a_0**2) / (2 * np.pi * c_0) * centi
        eigs = self.eig_mass_weight_hess()
        return np.sign(eigs) * np.sqrt(np.abs(eigs)) * coef

    def print_solution_02(self):
        print("=== Mass-Weighted Hessian Matrix (in unit Eh/(amu*a0^2)) ===")
        print(self.mass_weighted_hess())
        print("=== Eigenvalue of Mass-Weighted Hessian Matrix (in unit Eh/(amu*a0^2)) ===")
        print(self.eig_mass_weight_hess())
        print("=== Harmonic Vibrational Frequencies (in unit cm^-1) ===")
        print(self.harmonic_vib_freq())


if __name__ == '__main__':
    mol = Molecule()
    mol.construct_from_dat_file("input/h2o_geom.txt")
    mol.obtain_hessian("input/h2o_hessian.txt")
    mol.print_solution_02()
