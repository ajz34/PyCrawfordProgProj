from pyscf import gto
import numpy as np
import scipy.linalg
from scipy.linalg import fractional_matrix_power
from typing import Tuple
np.set_printoptions(precision=7, linewidth=120, suppress=True)


class Molecule:
    def __init__(self):
        # Project 03 existed
        self.atom_charges = NotImplemented  # type: np.ndarray
        self.atom_coords = NotImplemented  # type: np.ndarray
        self.natm = NotImplemented  # type: int
        self.mol = NotImplemented  # type: gto.Mole
        self.nao = NotImplemented  # type: int
        self.charge = 0  # type: int
        self.nocc = NotImplemented  # type: int
        # Project 04 added
        self.mo_coeff = NotImplemented  # type: np.ndarray
        self.mo_energy = NotImplemented  # type: np.ndarray
        self.eri_ao = NotImplemented  # type: np.ndarray
        self.eri_mo = NotImplemented  # type: np.ndarray
        self.energy_rhf = NotImplemented  # type: np.ndarray
        self.energy_corr = NotImplemented  # type: np.ndarray

    def construct_from_dat_file(self, file_path: str):
        # Same to Project 01
        with open(file_path, "r") as f:
            dat = np.array([line.split() for line in f.readlines()][1:])
            self.atom_charges = np.array(dat[:, 0], dtype=float).astype(int)
            self.atom_coords = np.array(dat[:, 1:4], dtype=float)
            self.natm = self.atom_charges.shape[0]

    def obtain_mol_instance(self, basis: str, verbose=0):
        # Same to Project 03
        mol = gto.Mole()
        mol.unit = "Bohr"
        mol.atom = "\n".join([("{:3d} " + " ".join(["{:25.18f}"] * 3)).format(chg, *coord) for chg, coord in zip(self.atom_charges, self.atom_coords)])
        mol.basis = basis
        mol.charge = self.charge
        mol.spin = 0
        mol.verbose = verbose
        self.mol = mol.build()

    def eng_nuclear_repulsion(self) -> float:
        # Same to Project 03, PySCF approach
        return self.mol.energy_nuc()

    def obtain_nao(self):
        # Same to Project 03, PySCF approach
        self.nao = self.mol.nao_nr()

    def obtain_nocc(self):
        # Same to Project 03
        assert (self.atom_charges.sum() - self.charge) % 2 == 0
        self.nocc = (self.atom_charges.sum() - self.charge) // 2

    def integral_ovlp(self) -> np.ndarray:
        # Same to Project 03, PySCF approach
        return self.mol.intor("int1e_ovlp")

    def integral_kin(self) -> np.ndarray:
        # Same to Project 03, PySCF approach
        return self.mol.intor("int1e_kin")

    def integral_nuc(self) -> np.ndarray:
        # Same to Project 03, PySCF approach
        return self.mol.intor("int1e_nuc")

    def get_hcore(self) -> np.ndarray:
        # Same to Project 03
        return self.integral_kin() + self.integral_nuc()

    def integral_eri(self) -> np.ndarray:
        # Same to Project 03, PySCF approach
        return self.mol.intor("int2e")

    def integral_ovlp_m1d2(self) -> np.ndarray:
        # Same to Project 03, SciPy approach
        return fractional_matrix_power(self.integral_ovlp(), -1/2)

    def obtain_eri_ao(self):
        # Attribute Modification: `eri_ao` atomic orbital electron repulsion integral
        self.eri_ao = self.integral_eri()

    def get_fock(self, dm: np.ndarray) -> np.ndarray:
        # Same to Project 03, but use self.eri_ao
        return self.get_hcore() + (self.eri_ao * dm).sum(axis=(-1, -2)) - 0.5 * (self.eri_ao * dm[:, None, :]).sum(axis=(-1, -3))

    def get_coeff_from_fock_diag(self, fock: np.ndarray) -> np.ndarray:
        # Same to Project 03, Scipy Approach
        return scipy.linalg.eigh(fock, self.integral_ovlp())[1]

    def make_rdm1(self, coeff: np.ndarray) -> np.ndarray:
        # Same to Project 03
        return 2 * coeff[:, :self.nocc] @ coeff[:, :self.nocc].T

    def get_updated_dm(self, dm: np.ndarray) -> np.ndarray:
        # Same to Project 03
        return self.make_rdm1(self.get_coeff_from_fock_diag(self.get_fock(dm)))

    def eng_total(self, dm: np.ndarray) -> float:
        # Same to Project 03
        return (0.5 * (self.get_hcore() + self.get_fock(dm)) * dm).sum() + self.eng_nuclear_repulsion()

    def scf_process(self, dm_guess: np.ndarray=None) -> Tuple[float, np.ndarray]:
        # Same to Project 03, but do not make debug output
        eng, dm = 0., np.zeros((self.nao, self.nao)) if dm_guess is None else np.copy(dm_guess)
        max_iter, thresh_eng, thresh_dm = 64, 1e-10, 1e-8
        for epoch in range(max_iter):
            eng_next, dm_next = self.eng_total(dm), self.get_updated_dm(dm)
            if np.abs(eng_next - eng) < thresh_eng and np.linalg.norm(dm_next - dm) < thresh_dm:
                eng, dm = eng_next, dm_next
                break
            eng, dm = eng_next, dm_next
        return eng, dm

    def obtain_scf_intermediates(self, dm_guess: np.ndarray=None):
        # Attribute Modification: `energy_rhf` Total energy of RHF
        # Attribute Modification: `mo_energy` Molecular orbital energies
        # Attribute Modification: `mo_coeff` Molecular orbital coefficients
        eng, dm = self.scf_process(dm_guess)
        self.energy_rhf = eng
        self.mo_energy, self.mo_coeff = scipy.linalg.eigh(self.get_fock(dm), self.integral_ovlp())

    @staticmethod
    def repeat_loop(dim: int, nested: int) -> np.ndarray:
        return np.array([np.tile(np.repeat(np.arange(dim), dim**(nested - 1 - i)), dim**i) for i in range(nested)]).T

    def get_eri_mo_naive(self) -> np.ndarray:
        # Naive algorithm
        # Output: MO electron repulsion integral
        eri_ao, coeff = self.eri_ao, self.mo_coeff
        loop_indices = self.repeat_loop(self.nao, 4)
        eri_mo = np.zeros((self.nao, self.nao, self.nao, self.nao))
        for p, q, r, s in loop_indices:
            for u, v, k, l in loop_indices:
                eri_mo[p, q, r, s] += eri_ao[u, v, k, l] * coeff[u, p] * coeff[v, q] * coeff[k, r] * coeff[l, s]
        return eri_mo

    def get_eri_mo_smarter(self):
        # Smarter algorithm
        # Output: MO electron repulsion integral
        eri_ao, coeff = self.eri_ao, self.mo_coeff
        loop_indices = self.repeat_loop(self.nao, 4)
        tmp_1 = np.zeros((self.nao, self.nao, self.nao, self.nao))
        for u, v, k, l in loop_indices:
            for p in range(self.nao):
                tmp_1[p, v, k, l] += eri_ao[u, v, k, l] * coeff[u, p]
        tmp_2 = np.zeros((self.nao, self.nao, self.nao, self.nao))
        for p, v, k, l in loop_indices:
            for q in range(self.nao):
                tmp_2[p, q, k, l] += tmp_1[p, v, k, l] * coeff[v, q]
        tmp_1.fill(0)
        for p, q, k, l in loop_indices:
            for r in range(self.nao):
                tmp_1[p, q, r, l] += tmp_2[p, q, k, l] * coeff[k, r]
        tmp_2.fill(0)
        for p, q, r, l in loop_indices:
            for s in range(self.nao):
                tmp_2[p, q, r, s] += tmp_1[p, q, r, l] * coeff[l, s]
        return tmp_2

    def get_eri_mo_einsum(self):
        # Use numpy.einsum to automatically find optimal contraction path
        # Output: MO electron repulsion integral
        return np.einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri_ao, self.mo_coeff, self.mo_coeff, self.mo_coeff, self.mo_coeff, optimize=True)

    def obtain_eri_mo(self):
        # Attribute Modification: `eri_mo` Molecular orbital electron repulsion integral
        self.eri_mo = self.get_eri_mo_einsum()

    def eng_mp2_corr(self):
        # Output: (Restricted) MP2 correlation energy
        nocc, e = self.nocc, self.mo_energy
        eri_iajb = self.eri_mo[:nocc, nocc:, :nocc, nocc:]
        D_iajb = e[:nocc, None, None, None] - e[None, nocc:, None, None] + e[None, None, :nocc, None] - e[None, None, None, nocc:]
        return (eri_iajb * (2 * eri_iajb - eri_iajb.swapaxes(-1, -3)) / D_iajb).sum()

    def obtain_mp2_corr(self):
        # Attribute Modification: `energy_corr` Post-HF (in this project is RMP2) correlation energy
        self.energy_corr = self.eng_mp2_corr()

    def print_solution_04(self):
        self.obtain_nao()
        self.obtain_nocc()
        self.obtain_eri_ao()
        self.obtain_scf_intermediates()
        self.obtain_eri_mo()
        self.obtain_mp2_corr()
        print("SCF total       energy: {:16.8f}".format(self.energy_rhf))
        print("MP2 correlation energy: {:16.8f}".format(self.energy_corr))
        print("MP2 total       energy: {:16.8f}".format(self.energy_rhf + self.energy_corr))


if __name__ == '__main__':
    sol_mole = Molecule()
    sol_mole.construct_from_dat_file("input/h2o/STO-3G/geom.dat")
    sol_mole.obtain_mol_instance(basis="STO-3G")
    sol_mole.print_solution_04()

