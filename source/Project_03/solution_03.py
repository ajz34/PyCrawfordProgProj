from pyscf import gto
import numpy as np
import scipy.linalg
from scipy.linalg import fractional_matrix_power
from typing import Dict, Tuple
np.set_printoptions(precision=7, linewidth=120, suppress=True)


class Molecule:
    def __init__(self):
        self.atom_charges = NotImplemented  # type: np.ndarray
        self.atom_coords = NotImplemented  # type: np.ndarray
        self.natm = NotImplemented  # type: int
        self.mol = NotImplemented  # type: gto.Mole
        self.path_dict = NotImplemented  # type: Dict[str, str]
        self.nao = NotImplemented  # type: int
        self.charge = 0  # type: int
        self.nocc = NotImplemented  # type: int

    def construct_from_dat_file(self, file_path: str):
        # Same to Project 01
        with open(file_path, "r") as f:
            dat = np.array([line.split() for line in f.readlines()][1:])
            self.atom_charges = np.array(dat[:, 0], dtype=float).astype(int)
            self.atom_coords = np.array(dat[:, 1:4], dtype=float)
            self.natm = self.atom_charges.shape[0]

    def obtain_mol_instance(self, basis: str, verbose=0):
        # Input: Basis string information
        #        (for advanced users, see PySCF API, this could be dictionary instead of string)
        # Attribute modification: Obtain `pyscf.gto.Mole` instance `self.mol`
        mol = gto.Mole()
        mol.unit = "Bohr"
        mol.atom = "\n".join([("{:3d} " + " ".join(["{:25.18f}"] * 3)).format(chg, *coord) for chg, coord in zip(self.atom_charges, self.atom_coords)])
        mol.basis = basis
        mol.charge = self.charge
        mol.spin = 0
        mol.verbose = verbose
        self.mol = mol.build()

    def eng_nuclear_repulsion(self) -> float:
        # Output: Nuclear Repulsion Energy
        def approach_pyscf():
            return self.mol.energy_nuc()

        def approach_input():
            with open(self.path_dict["enuc"], "r") as f:
                return float(f.read().split()[0])

        def approach_direct():
            rinv = 1 / (np.linalg.norm(self.atom_coords[:, None, :] - self.atom_coords[None, :, :], axis=-1) + np.diag(np.ones(self.natm) * np.inf))
            return (0.5 * self.atom_charges[:, None] * self.atom_charges[None, :] * rinv).sum()

        return approach_pyscf()

    def obtain_nao(self, file_path: str=NotImplemented):
        # Input: Any integral file (only used in read-input approach, pyscf approach could simply leave NotImplemented)
        # Attribute Modification: `nao` number of atomic orbitals
        if self.mol is not NotImplemented:
            self.nao = self.mol.nao_nr()
            return
        assert file_path is not NotImplemented
        with open(file_path, "r") as f:
            self.nao = int(f.readlines()[-1].split()[0])

    def obtain_nocc(self):
        # Attribute Modification: `nocc` occupied orbital
        assert (self.atom_charges.sum() - self.charge) % 2 == 0
        self.nocc = (self.atom_charges.sum() - self.charge) // 2

    def read_integral_1e(self, file_path: str) -> np.ndarray:
        # Input: Integral file
        # Output: Symmetric integral matrix
        res = np.zeros((self.nao, self.nao))
        with open(file_path, "r") as f:
            for line in f.readlines():
                i, j, val = line.split()
                i, j, val = int(i) - 1, int(j) - 1, float(val)
                res[i][j] = res[j][i] = val
        return res

    def integral_ovlp(self) -> np.ndarray:
        # Output: Overlap
        def approach_pyscf():
            return self.mol.intor("int1e_ovlp")

        def approach_input():
            return self.read_integral_1e(self.path_dict["s"])

        return approach_pyscf()

    def integral_kin(self) -> np.ndarray:
        # Output: Kinetic Integral
        def approach_pyscf():
            return self.mol.intor("int1e_kin")

        def approach_input():
            return self.read_integral_1e(self.path_dict["t"])

        return approach_pyscf()

    def integral_nuc(self) -> np.ndarray:
        # Output: Nuclear Attraction Integral
        def approach_pyscf():
            return self.mol.intor("int1e_nuc")

        def approach_input():
            return self.read_integral_1e(self.path_dict["v"])

        return approach_pyscf()

    def get_hcore(self) -> np.ndarray:
        # Output: Hamiltonian Core
        return self.integral_kin() + self.integral_nuc()

    def read_integral_2e(self, file_path: str) -> np.ndarray:
        # Input: Integral file
        # Output: 8-fold symmetric 2-e integral tensor
        res = np.zeros((self.nao, self.nao, self.nao, self.nao))
        with open(file_path, "r") as f:
            for line in f.readlines():
                i, j, k, l, val = line.split()
                i, j, k, l, val = int(i) - 1, int(j) - 1, int(k) - 1, int(l) - 1, float(val)
                res[i][j][k][l] = res[i][j][l][k] = res[j][i][k][l] = res[j][i][l][k] \
              = res[k][l][i][j] = res[l][k][i][j] = res[k][l][j][i] = res[l][k][j][i] = val
        return res

    def integral_eri(self) -> np.ndarray:
        # Output: Electron repulsion integral
        def approach_pyscf():
            return self.mol.intor("int2e")

        def approach_input():
            return self.read_integral_2e(self.path_dict["eri"])

        return approach_pyscf()

    def integral_ovlp_m1d2(self) -> np.ndarray:
        # Output: S^{-1/2} in symmetric orthogonalization
        def approach_scipy():
            return fractional_matrix_power(self.integral_ovlp(), -1/2)

        def approach_eigen():
            eig, l = np.linalg.eigh(self.integral_ovlp())
            return l @ np.diag(eig**(-1/2)) @ l.T

        return approach_eigen()

    def get_fock(self, dm: np.ndarray) -> np.ndarray:
        # Input: Density matrix
        # Output: Fock matrix (based on input density matrix, not only final fock matrix)
        eri = self.integral_eri()
        return self.get_hcore() + (eri * dm).sum(axis=(-1, -2)) - 0.5 * (eri * dm[:, None, :]).sum(axis=(-1, -3))

    def get_coeff_from_fock_diag(self, fock: np.ndarray) -> np.ndarray:
        # Input: Fock matrix
        # Output: Orbital coefficient obtained by fock matrix diagonalization
        # Conventional approach
        # fock = self.get_fock(dm)
        # ovlp_m1d2 = self.integral_ovlp_m1d2()
        # orth_fock = ovlp_m1d2 @ fock @ ovlp_m1d2
        # orth_coeff = np.linalg.eigh(orth_fock)[1]
        # coeff = ovlp_m1d2 @ orth_coeff
        # return coeff
        # SciPy approach
        return scipy.linalg.eigh(fock, self.integral_ovlp())[1]

    def make_rdm1(self, coeff: np.ndarray) -> np.ndarray:
        # Input: molecular orbital coefficient
        # Output: density for the given orbital coefficient
        return 2 * coeff[:, :self.nocc] @ coeff[:, :self.nocc].T

    def get_updated_dm(self, dm: np.ndarray) -> np.ndarray:
        # Input: Guess density matrix
        # Output: Updated density matrix
        fock = self.get_fock(dm)
        coeff = self.get_coeff_from_fock_diag(fock)
        return self.make_rdm1(coeff)

    def eng_total(self, dm: np.ndarray) -> float:
        # Input: Any density matrix (although one should impose tr(D@S)=nelec to satisfy variational principle)
        # Output: SCF energy (electronic contribution + nuclear repulsion contribution)
        # For reproduction of original project output, code should be following:
        # dm_new = self.get_updated_dm(dm)
        # return (0.5 * (self.get_hcore() + self.get_fock(dm)) * dm_new).sum()
        return (0.5 * (self.get_hcore() + self.get_fock(dm)) * dm).sum() + self.eng_nuclear_repulsion()

    def scf_process(self, dm_guess: np.ndarray=None) -> Tuple[float, np.ndarray]:
        # Input: Density matrix guess
        # Output: Converged SCF total energy and density matrix
        eng, dm = 0., np.zeros((self.nao, self.nao)) if dm_guess is None else np.copy(dm_guess)
        eng_next, dm_next = NotImplemented, NotImplemented
        max_iter, thresh_eng, thresh_dm = 64, 1e-10, 1e-8
        print("{:>5} {:>20} {:>20} {:>20}".format("Epoch", "Total Energy", "Energy Deviation", "Density Deviation"))
        for epoch in range(max_iter):
            eng_next = self.eng_total(dm)
            dm_next = self.get_updated_dm(dm)
            print("{:5d} {:20.12f} {:20.12f} {:20.12f}".format(epoch, eng_next, eng_next - eng, np.linalg.norm(dm_next - dm)))
            if np.abs(eng_next - eng) < thresh_eng and np.linalg.norm(dm_next - dm) < thresh_dm:
                break
            eng, dm = eng_next, dm_next
        return eng, dm

    def integral_dipole(self) -> np.ndarray:
        # Output: Dipole related integral (mu|r|nu) in dimension (3, nao, nao)
        def approach_pyscf():
            return self.mol.intor("int1e_r")

        def approach_input():
            return np.array([
                self.read_integral_1e(self.path_dict["mux"]),
                self.read_integral_1e(self.path_dict["muy"]),
                self.read_integral_1e(self.path_dict["muz"]),
            ])

        return approach_pyscf()

    def get_dipole(self, dm: np.array) -> np.ndarray:
        # Input: SCF converged density matrix
        # Output: Molecule dipole in A.U.
        return - (self.integral_dipole() * dm).sum(axis=(-1, -2)) + (self.atom_charges[:, None] * self.atom_coords).sum(axis=0)

    def population_analysis(self, dm: np.array) -> np.ndarray:
        # Input: SCF converged density matrix
        # Output: Mulliken population analysis, charges on every atom
        pop_charges = np.zeros(self.natm)
        dm_ovlp = dm @ self.integral_ovlp()
        for A in range(self.natm):
            _, _, p0, p1 = self.mol.aoslice_by_atom()[A]
            pop_charges[A] = self.atom_charges[A] - dm_ovlp[p0:p1, p0:p1].trace()
        return pop_charges

    def print_solution_03(self):
        print("=== SCF Process ===")
        dm_guess = np.zeros((self.nao, self.nao))
        _, dm_converged = self.scf_process(dm_guess)
        print("=== Dipole (in A.U.) ===")
        print(self.get_dipole(dm_converged))
        print("=== Mulliken Population Analysis ===")
        print(self.population_analysis(dm_converged))


if __name__ == '__main__':
    sol_mole = Molecule()
    sol_mole.construct_from_dat_file("input/h2o/STO-3G/geom.dat")
    sol_mole.obtain_mol_instance(basis="STO-3G")
    sol_mole.obtain_nocc()
    sol_mole.path_dict = {
        "enuc": "input/h2o/STO-3G/enuc.dat",
        "s": "input/h2o/STO-3G/s.dat",
        "t": "input/h2o/STO-3G/t.dat",
        "v": "input/h2o/STO-3G/v.dat",
        "eri": "input/h2o/STO-3G/eri.dat",
        "mux": "input/h2o/STO-3G/mux.dat",
        "muy": "input/h2o/STO-3G/muy.dat",
        "muz": "input/h2o/STO-3G/muz.dat",
    }
    sol_mole.obtain_nao(file_path=sol_mole.path_dict["s"])
    sol_mole.print_solution_03()

