import numpy as np
from molmass import ELEMENTS
from scipy import constants
from scipy.constants import physical_constants


class Molecule:

    def __init__(self):
        self.atom_charges = NotImplemented  # type: np.ndarray
        self.atom_coords = NotImplemented  # type: np.ndarray
        self.natm = NotImplemented  # type: int

    def construct_from_dat_file(self, file_path: str):
        # Input: Read file from `file_path`
        # Attribute modification: Obtain atom charges to `self.atom_charges`
        # Attribute modification: Obtain coordinates to `self.atom_coords`
        # Attribute modification: Obtain atom numbers in molecule to `self.natm`
        with open(file_path, "r") as f:
            # Ignore the redundant first line
            dat = np.array([line.split() for line in f.readlines()][1:])
            self.atom_charges = np.array(dat[:, 0], dtype=int)
            self.atom_coords = np.array(dat[:, 1:4], dtype=float)
            self.natm = self.atom_charges.shape[0]

    def bond_length(self, i: int, j: int) -> float:
        # Input: `i`, `j` index of molecule's atom
        # Output: Bond length from atom `i` to atom `j`
        return np.linalg.norm(self.atom_coords[i] - self.atom_coords[j])

    def print_bond_length(self):
        # Usage: Print all bond length
        print("=== Bond Length ===")
        for i in range(self.natm):
            for j in range(i + 1, self.natm):
                print("{:3d} - {:3d}: {:10.5f} Bohr".format(i, j, self.bond_length(i, j)))

    def bond_unit_vector(self, i: int, j: int) -> np.ndarray:
        # Input: `i`, `j` index of molecule's atom
        # Output: Unit vector of bond from atom `i` to atom `j`
        vec = self.atom_coords[j] - self.atom_coords[i]
        return vec / np.linalg.norm(vec)

    def bond_angle(self, i: int, j: int, k: int) -> float:
        # Input: `i`, `j`, `k` index of molecule's atom; where `j` is the central atom
        # Output: Bond angle for atoms `i`-`j`-`k`
        e_ji = self.bond_unit_vector(j, i)
        e_jk = self.bond_unit_vector(j, k)
        return np.arccos(e_ji.dot(e_jk)) * 180 / np.pi

    def is_valid_bond_angle(self, i: int, j: int, k: int) -> bool:
        # Input: `i`, `j`, `k` index of molecule's atom; where `j` is the central atom
        # Output: Test if `i`-`j`-`k` is a valid bond angle
        #         if i != j != k
        #         and if i-j and j-k bond length smaller than 3 Angstrom,
        #         and if angle i-j-k > 90 degree
        return len({i, j, k}) == 3 and self.bond_length(i, j) < 3 and self.bond_length(j, k) < 3 and self.bond_angle(i, j, k) > 90

    def print_bond_angle(self):
        # Usage: Print all bond angle i-j-k which is considered as valid angle
        print("=== Bond Angle ===")
        for i in range(self.natm):
            for j in range(i + 1, self.natm):
                for k in range(j + 1, self.natm):
                    for tup in [(i, j, k), (j, i, k), (i, k, j)]:
                        if self.is_valid_bond_angle(*tup):
                            print("{:3d} - {:3d} - {:3d}: {:10.5f} Degree".format(*tup, self.bond_angle(*tup)))
                            break

    def out_of_plane_angle(self, i: int, j: int, k: int, l: int) -> float:
        # Input: `i`, `j`, `k`, `l` index of molecule's atom; where `k` is the central atom, and angle is i - j-k-l
        # Output: Out-of-plane bond angle for atoms `i`-`j`-`k`-`l`
        res = np.cross(self.bond_unit_vector(k, j), self.bond_unit_vector(k, l)).dot(self.bond_unit_vector(k, i))
        res /= np.sin(self.bond_angle(j, k, l) / 180 * np.pi)
        assert(np.abs(res) < 1 + 1e-7)
        res = np.sign(res) if np.abs(res) > 1 else res
        return np.arcsin(res) * 180 / np.pi

    def is_valid_out_of_plane_angle(self, i: int, j: int, k: int, l: int) -> bool:
        # Input: `i`, `j`, `k`, `l` index of molecule's atom; where `k` is the central atom, and angle is i - j-k-l
        # Output: Test if `i`-`j`-`k` is a valid out-of-plane bond angle
        #         if i != j != k != l
        #         and if angle j-k-l is valid bond angle
        #         and if i-k bond length smaller than 3 Angstrom
        #         and bond angle of j-k-l is not linear
        return len({i, j, k, l}) == 4 and self.is_valid_bond_angle(j, k, l) and self.bond_length(k, i) < 3 and self.bond_angle(j, k, l) < 179

    def print_out_of_plane_angle(self):
        # Usage: Print all out-of-plane bond angle i-j-k-l which is considered as valid
        print("=== Out-of-Plane Angle ===")
        for j in range(self.natm):
            for k in range(j + 1, self.natm):
                for l in range(k + 1, self.natm):
                    for tup in [(j, k, l), (k, j, l), (j, l, k)]:
                        if self.is_valid_bond_angle(*tup):
                            for i in range(self.natm):
                                if i not in [j, k, l] and self.is_valid_out_of_plane_angle(i, *tup):
                                    print("{:3d} - {:3d} - {:3d} - {:3d}: {:10.5f} Degree".format(i, *tup, self.out_of_plane_angle(i, *tup)))

    def dihedral_angle(self, i: int, j: int, k: int, l: int) -> float:
        # Input: `i`, `j`, `k`, `l` index of molecule's atom; where `k` is the central atom, and angle is i - j-k-l
        # Output: Dihedral angle for atoms `i`-`j`-`k`-`l`
        res = np.cross(self.bond_unit_vector(j, i), self.bond_unit_vector(j, k)).dot(np.cross(self.bond_unit_vector(k, j), self.bond_unit_vector(k, l)))
        res /= np.sin(self.bond_angle(i, j, k) / 180 * np.pi) * np.sin(self.bond_angle(j, k, l) / 180 * np.pi)
        assert(np.abs(res) < 1 + 1e-7)
        res = np.sign(res) if np.abs(res) > 1 else res
        return np.arccos(res) * 180 / np.pi

    def is_valid_dihedral_angle(self, i: int, j: int, k: int, l: int) -> bool:
        # Input: `i`, `j`, `k`, `l` index of molecule's atom; where `k` is the central atom, and angle is i - j-k-l
        # Output: Test if `i`-`j`-`k` is a valid dihedral bond angle
        #         if i != j != k != l
        #         and if i, j, k construct a valid bond angle (with j-k bonded)
        #         and if j, k, l construct a valid bond angle (with j-k bonded)
        return len({i, j, k, l}) == 4 \
               and (self.is_valid_bond_angle(i, j, k) or self.is_valid_bond_angle(i, k, j)) \
               and (self.is_valid_bond_angle(j, k, l) or self.is_valid_bond_angle(k, j, l))

    def print_dihedral_angle(self):
        # Usage: Print all dihedral bond angle i-j-k-l which is considered as valid
        print("=== Dihedral Angle ===")
        for j in range(self.natm):
            for k in range(j + 1, self.natm):
                for i in range(self.natm):
                    for l in range(i + 1, self.natm):
                        if self.is_valid_dihedral_angle(i, j, k, l):
                            print("{:3d} - {:3d} - {:3d} - {:3d}: {:10.5f} Degree".format(i, j, k, l, self.dihedral_angle(i, j, k, l)))

    def center_of_mass(self) -> np.ndarray or list:
        # Output: Center of mass for this molecule
        sum_of_vec, sum_of_mass = np.zeros(3), 0
        for i in range(self.natm):
            sum_of_vec += ELEMENTS[self.atom_charges[i]].mass * self.atom_coords[i]
            sum_of_mass += ELEMENTS[self.atom_charges[i]].mass
        return sum_of_vec / sum_of_mass

    def moment_of_inertia(self):
        # Output: Principle of moments of intertia
        atom_weights = np.array([ELEMENTS[c].mass for c in self.atom_charges])
        trans_coords = self.atom_coords - self.center_of_mass()
        res = - np.einsum("i, ix, iy -> xy", atom_weights, trans_coords, trans_coords)
        np.fill_diagonal(res, res.diagonal() - res.diagonal().sum())
        return np.linalg.eigvalsh(res)

    def print_moment_of_interia(self):
        # Output: Print moments of inertia in amu bohr2, amu Å2, and g cm2
        moment_in_amu_bohr = self.moment_of_inertia()
        print("In {:>15}: {:16.8e} {:16.8e} {:16.8e}".format(
            "amu bohr^2", *(moment_in_amu_bohr)))
        print("In {:>15}: {:16.8e} {:16.8e} {:16.8e}".format(
            "amu angstrom^2", *(moment_in_amu_bohr * physical_constants["Bohr radius"][0]**2 / constants.angstrom**2)))
        print("In {:>15}: {:16.8e} {:16.8e} {:16.8e}".format(
            "g cm^2", *(moment_in_amu_bohr * physical_constants["atomic mass constant"][0] * constants.kilo * physical_constants["Bohr radius"][0]**2 / constants.centi**2)))

    def type_of_moment_of_interia(self) -> str:
        # Output: Judge which type of moment of interia is
        m = self.moment_of_inertia()
        if np.abs(m[0] - m[1]) < 1e-4:
            return "Spherical" if np.abs(m[1] - m[2]) < 1e-4 else "Oblate"
        elif np.abs(m[0]) < 1e-4 or np.abs((m[1] - m[0]) / m[0]) > 1e4:
            return "Linear"
        else:
            return "Prolate" if np.abs(m[1] - m[2]) < 1e-4 else "Asymmetric"

    def rotational_constants(self) -> np.ndarray or list:
        # Output: Rotational constant in cm^-1
        return constants.Planck / (8 * np.pi**2 * constants.c * self.moment_of_inertia()) / physical_constants["atomic mass constant"][0] / physical_constants["Bohr radius"][0]**2 * constants.centi

    def print_solution_01(self):
        print("=== Atom Charges ===")
        print(self.atom_charges)
        print("=== Coordinates ===")
        print(self.atom_coords)
        self.print_bond_length()
        self.print_bond_angle()
        self.print_out_of_plane_angle()
        self.print_dihedral_angle()
        print("=== Center of Mass ===")
        print("{:10.5f} {:10.5f} {:10.5f}".format(*self.center_of_mass()))
        print("=== Moments of Inertia ===")
        self.print_moment_of_interia()
        print("Type: ", self.type_of_moment_of_interia())
        print("=== Rotational Constants ===")
        print("In cm^-1: {:12.5f} {:12.5f} {:12.5f}".format(*self.rotational_constants()))
        print("In   MHz: {:12.5f} {:12.5f} {:12.5f}".format(*(self.rotational_constants() * constants.c / constants.centi / constants.mega)))
