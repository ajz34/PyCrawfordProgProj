from pyscf import gto
import numpy as np
import scipy.linalg
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
        # Project 05 added
        self.w = NotImplemented  # type: np.ndarray
        self.w_ovoo = NotImplemented  # type: np.ndarray
        self.w_ovvo = NotImplemented  # type: np.ndarray
        self.w_ovov = NotImplemented  # type: np.ndarray
        self.w_ovvv = NotImplemented  # type: np.ndarray
        self.v_oooo = NotImplemented  # type: np.ndarray
        self.v_ovoo = NotImplemented  # type: np.ndarray
        self.v_ovvo = NotImplemented  # type: np.ndarray
        self.v_ovov = NotImplemented  # type: np.ndarray
        self.v_oovv = NotImplemented  # type: np.ndarray
        self.v_ovvv = NotImplemented  # type: np.ndarray
        self.v_vvvv = NotImplemented  # type: np.ndarray

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

    def obtain_nao(self):
        # Same to Project 03, PySCF approach
        self.nao = self.mol.nao_nr()

    def obtain_nocc(self):
        # Same to Project 03
        assert (self.atom_charges.sum() - self.charge) % 2 == 0
        self.nocc = (self.atom_charges.sum() - self.charge) // 2

    def obtain_eri_ao(self):
        # Same to Project 04
        self.eri_ao = self.mol.intor("int2e")

    def get_hcore(self) -> np.ndarray:
        # Same to Project 03
        return self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc")

    def get_fock(self, dm: np.ndarray) -> np.ndarray:
        # Same to Project 03, but use self.eri_ao
        return self.get_hcore() + (self.eri_ao * dm).sum(axis=(-1, -2)) - 0.5 * (self.eri_ao * dm[:, None, :]).sum(axis=(-1, -3))

    def make_rdm1(self, coeff: np.ndarray) -> np.ndarray:
        # Same to Project 03
        return 2 * coeff[:, :self.nocc] @ coeff[:, :self.nocc].T

    def eng_total(self, dm: np.ndarray) -> float:
        # Same to Project 03
        return (0.5 * (self.get_hcore() + self.get_fock(dm)) * dm).sum() + self.mol.energy_nuc()

    def scf_process(self, dm_guess: np.ndarray=None) -> Tuple[float, np.ndarray]:
        # Same to Project 03, but do not make debug output
        eng, dm = 0., np.zeros((self.nao, self.nao)) if dm_guess is None else np.copy(dm_guess)
        max_iter, thresh_eng, thresh_dm = 64, 1e-10, 1e-8
        for epoch in range(max_iter):
            eng_next = self.eng_total(dm)
            dm_next = self.make_rdm1(scipy.linalg.eigh(self.get_fock(dm), self.mol.intor("int1e_ovlp"))[1])
            if np.abs(eng_next - eng) < thresh_eng and np.linalg.norm(dm_next - dm) < thresh_dm:
                eng, dm = eng_next, dm_next
                break
            eng, dm = eng_next, dm_next
        return eng, dm

    def obtain_scf_intermediates(self, dm_guess: np.ndarray=None):
        # Same to Project 04
        eng, dm = self.scf_process(dm_guess)
        self.energy_rhf = eng
        self.mo_energy, self.mo_coeff = scipy.linalg.eigh(self.get_fock(dm), self.mol.intor("int1e_ovlp"))

    def get_eri_mo_einsum(self):
        # Same to Project 04
        return np.einsum("uvkl, up, vq, kr, ls -> pqrs", self.eri_ao, self.mo_coeff, self.mo_coeff, self.mo_coeff, self.mo_coeff, optimize=True)

    def obtain_eri_mo(self):
        # Same to Project 04
        self.eri_mo = self.get_eri_mo_einsum()

    def obtain_w(self):
        # Attribute Modification: `w` biorthogonal electron repulsion integral
        self.w = 2 * self.eri_mo - self.eri_mo.swapaxes(1, 3)

    def obtain_wv_slices(self):
        # Attribute Modification: Various (biorthogonal) ERI slices
        nocc, nmo = self.nocc, self.nao
        so, sv = slice(0, nocc), slice(nocc, nmo)
        self.w_ovoo = self.w[so, sv, so, so]
        self.w_ovvo = self.w[so, sv, sv, so]
        self.w_ovov = self.w[so, sv, so, sv]
        self.w_ovvv = self.w[so, sv, sv, sv]
        self.v_oooo = self.eri_mo[so, so, so, so]
        self.v_ovoo = self.eri_mo[so, sv, so, so]
        self.v_ovvo = self.eri_mo[so, sv, sv, so]
        self.v_ovov = self.eri_mo[so, sv, so, sv]
        self.v_oovv = self.eri_mo[so, so, sv, sv]
        self.v_ovvv = self.eri_mo[so, sv, sv, sv]
        self.v_vvvv = self.eri_mo[sv, sv, sv, sv]

    def get_t1_t2_initial_guess(self) -> Tuple[np.ndarray, np.ndarray]:
        # Output: `t1`, `t2` Initial guess of CCSD amplitudes
        nocc, nmo, e, eri_mo = self.nocc, self.nao, self.mo_energy, self.eri_mo
        t1 = np.zeros((nocc, nmo - nocc))
        D_ijab = e[:nocc, None, None, None] + e[None, :nocc, None, None] - e[None, None, nocc:, None] - e[None, None, None, nocc:]
        t2 = self.v_ovov.swapaxes(1, 2) / D_ijab
        return t1, t2

    def cc_Foo(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Ref: Hitara, eq 37
        Fki = np.diag(self.mo_energy[:self.nocc])
        Fki += np.einsum("kcld, ilcd   -> ki", self.w_ovov, t2,     optimize=True)
        Fki += np.einsum("kcld, ic, ld -> ki", self.w_ovov, t1, t1, optimize=True)
        return Fki

    def cc_Fvv(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Ref: Hitara, eq 38
        F_ac = np.diag(self.mo_energy[self.nocc:])
        F_ac -= np.einsum("kcld, klad   -> ac", self.w_ovov, t2,     optimize=True)
        F_ac -= np.einsum("kcld, ka, ld -> ac", self.w_ovov, t1, t1, optimize=True)
        return F_ac

    def cc_Fov(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Ref: Hitara, eq 39
        # Note that amplitude t2 is actually not taken into account,
        # but for signature consistency, we still include this useless amplitude
        return np.einsum("kcld, ld -> kc", self.w_ovov, t1, optimize=True)

    def Loo(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Ref: Hitara, eq 40
        L_ki = self.cc_Foo(t1, t2)
        L_ki += np.einsum("lcki, lc -> ki", self.w_ovoo, t1, optimize=True)
        return L_ki

    def Lvv(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Ref: Hitara, eq 41
        Lac = self.cc_Fvv(t1, t2)
        Lac += np.einsum("kdac, kd -> ac", self.w_ovvv, t1, optimize=True)
        return Lac

    def cc_Woooo(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Ref: Hitara, eq 42
        W_klij = np.copy(self.v_oooo).transpose((0, 2, 1, 3))
        W_klij += np.einsum("lcki, jc     -> klij", self.v_ovoo, t1,     optimize=True)
        W_klij += np.einsum("kclj, ic     -> klij", self.v_ovoo, t1,     optimize=True)
        W_klij += np.einsum("kcld, ijcd   -> klij", self.v_ovov, t2,     optimize=True)
        W_klij += np.einsum("kcld, ic, jd -> klij", self.v_ovov, t1, t1, optimize=True)
        return W_klij

    def cc_Wvvvv(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Ref: Hitara, eq 43
        W_abcd = np.copy(self.v_vvvv.transpose((0, 2, 1, 3)))
        W_abcd -= np.einsum("kdac, kb -> abcd", self.v_ovvv, t1, optimize=True)
        W_abcd -= np.einsum("kcbd, ka -> abcd", self.v_ovvv, t1, optimize=True)
        return W_abcd

    def cc_Wvoov(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Ref: Hitara, eq 44
        W_akic = np.copy(self.v_ovvo.transpose((2, 0, 3, 1)))
        W_akic -=       np.einsum("kcli, la     -> akic", self.v_ovoo, t1,     optimize=True)
        W_akic +=       np.einsum("kcad, id     -> akic", self.v_ovvv, t1,     optimize=True)
        W_akic -= 0.5 * np.einsum("ldkc, ilda   -> akic", self.v_ovov, t2,     optimize=True)
        W_akic -=       np.einsum("ldkc, id, la -> akic", self.v_ovov, t1, t1, optimize=True)
        W_akic += 0.5 * np.einsum("ldkc, ilad   -> akic", self.w_ovov, t2,     optimize=True)
        return W_akic

    def cc_Wvovo(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Ref: Hitara, eq 45
        Wakci = np.copy(self.v_oovv.transpose((2, 0, 3, 1)))
        Wakci -=       np.einsum("lcki, la     -> akci", self.v_ovoo, t1,     optimize=True)
        Wakci +=       np.einsum("kdac, id     -> akci", self.v_ovvv, t1,     optimize=True)
        Wakci -= 0.5 * np.einsum("lckd, ilda   -> akci", self.v_ovov, t2,     optimize=True)
        Wakci -=       np.einsum("lckd, id, la -> akci", self.v_ovov, t1, t1, optimize=True)
        return Wakci

    def update_t1(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Input: `t1`, `t2` CCSD amplitudes guess
        # Output: Updated `t1` CCSD single amplitude
        nocc = self.nocc
        Foo, Fvv, Fov = self.cc_Foo(t1, t2), self.cc_Fvv(t1, t2), self.cc_Fov(t1, t2)
        Foot = Foo - np.diag(self.mo_energy[:nocc])
        Fvvt = Fvv - np.diag(self.mo_energy[nocc:])

        RHS  =     np.einsum("ac, ic     -> ia", Fvvt, t1   , optimize=True)
        RHS -=     np.einsum("ki, ka     -> ia", Foot, t1   , optimize=True)
        RHS += 2 * np.einsum("kc, kica   -> ia", Fov, t2    , optimize=True)
        RHS -=     np.einsum("kc, ikca   -> ia", Fov, t2    , optimize=True)
        RHS +=     np.einsum("kc, ic, ka -> ia", Fov, t1, t1, optimize=True)
        RHS +=     np.einsum("kcai, kc     ->ia", self.w_ovvo, t1,     optimize=True)
        RHS +=     np.einsum("kdac, ikcd   ->ia", self.w_ovvv, t2,     optimize=True)
        RHS +=     np.einsum("kdac, kd, ic ->ia", self.w_ovvv, t1, t1, optimize=True)
        RHS -=     np.einsum("lcki, klac   ->ia", self.w_ovoo, t2,     optimize=True)
        RHS -=     np.einsum("lcki, lc, ka ->ia", self.w_ovoo, t1, t1, optimize=True)

        D_ov = self.mo_energy[:nocc, None] - self.mo_energy[None, nocc:]
        return RHS / D_ov

    def update_t2(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        # Input: `t1`, `t2` CCSD amplitudes guess
        # Output: Updated `t2` CCSD double amplitude
        nocc, e = self.nocc, self.mo_energy
        Loot = self.Loo(t1, t2) - np.diag(self.mo_energy[:nocc])
        Lvvt = self.Lvv(t1, t2) - np.diag(self.mo_energy[nocc:])
        Woooo = self.cc_Woooo(t1, t2)
        Wvoov = self.cc_Wvoov(t1, t2)
        Wvovo = self.cc_Wvovo(t1, t2)
        Wvvvv = self.cc_Wvvvv(t1, t2)

        RHS = 0.5 * np.copy(self.v_ovov.swapaxes(1, 2))
        RHS += 0.5 * np.einsum("klij, klab   -> ijab", Woooo, t2,     optimize=True)
        RHS += 0.5 * np.einsum("klij, ka, lb -> ijab", Woooo, t1, t1, optimize=True)
        RHS += 0.5 * np.einsum("abcd, ijcd   -> ijab", Wvvvv, t2,     optimize=True)
        RHS += 0.5 * np.einsum("abcd, ic, jd -> ijab", Wvvvv, t1, t1, optimize=True)

        RHS += np.einsum("ac,ijcb->ijab", Lvvt, t2, optimize=True)
        RHS -= np.einsum("ki,kjab->ijab", Loot, t2, optimize=True)

        RHS += np.einsum("iacb, jc     -> ijab", self.v_ovvv, t1,     optimize=True)
        RHS -= np.einsum("kibc, ka, jc -> ijab", self.v_oovv, t1, t1, optimize=True)
        RHS -= np.einsum("iajk, kb     -> ijab", self.v_ovoo, t1,     optimize=True)
        RHS -= np.einsum("iack, jc, kb -> ijab", self.v_ovvo, t1, t1, optimize=True)

        RHS += 2 * np.einsum("akic, kjcb -> ijab", Wvoov, t2, optimize=True)
        RHS -=     np.einsum("akci, kjcb -> ijab", Wvovo, t2, optimize=True)
        RHS -=     np.einsum("akic, kjbc -> ijab", Wvoov, t2, optimize=True)
        RHS -=     np.einsum("bkci, kjac -> ijab", Wvovo, t2, optimize=True)

        D_oovv = e[:nocc, None, None, None] + e[None, :nocc, None, None] - e[None, None, nocc:, None] - e[None, None, None, nocc:]
        return (RHS + RHS.transpose((1, 0, 3, 2))) / D_oovv

    def eng_ccsd_corr(self, t1: np.ndarray, t2: np.ndarray) -> float:
        # Input: `t1`, `t2` CCSD amplitudes
        # Output: (Closed-shell) CCSD correlation energy for given amplitudes (not converged value)
        eng  = np.einsum("iajb, ijab   -> ", self.w_ovov, t2,     optimize=True)
        eng += np.einsum("iajb, ia, jb -> ", self.w_ovov, t1, t1, optimize=True)
        return float(eng)  # add float to mute IDE strong warning info

    def ccsd_process(self) -> Tuple[float, np.ndarray, np.ndarray]:
        # Output: Converged CCSD correlation energy, and density matrix
        t1, t2 = self.get_t1_t2_initial_guess()
        eng_ccsd_corr, eng_ccsd_corr_new = 0, NotImplemented
        maxiter, thresh = 128, 1e-10
        print("{:>5} {:>20}".format("Epoch", "CCSD Corr Energy"))
        for epoch in range(maxiter):
            t1, t2 = self.update_t1(t1, t2), self.update_t2(t1, t2)
            eng_ccsd_corr_new = self.eng_ccsd_corr(t1, t2)
            print("{:5d} {:20.12f}".format(epoch, eng_ccsd_corr_new))
            if np.abs(eng_ccsd_corr_new - eng_ccsd_corr) < thresh:
                break
            eng_ccsd_corr = eng_ccsd_corr_new
        return eng_ccsd_corr_new, t1, t2

    def print_solution_05(self):
        self.obtain_nao()
        self.obtain_nocc()
        self.obtain_eri_ao()
        self.obtain_scf_intermediates()
        self.obtain_eri_mo()
        self.obtain_w()
        self.obtain_wv_slices()
        t1_guess, t2_guess = self.get_t1_t2_initial_guess()
        eng_mp2_corr = self.eng_ccsd_corr(t1_guess, t2_guess)
        print("=== CCSD Iterations ===")
        eng_ccsd_corr, t1, t2 = self.ccsd_process()
        print("=== Final Results ===")
        print(" MP2 Correlation energy: {:20.12f}".format(eng_mp2_corr))
        print("CCSD Correlation energy: {:20.12f}".format(eng_ccsd_corr))
        print(" SCF       Total energy: {:20.12f}".format(self.energy_rhf))
        print(" MP2       Total energy: {:20.12f}".format(self.energy_rhf + eng_mp2_corr))
        print("CCSD       Total energy: {:20.12f}".format(self.energy_rhf + eng_ccsd_corr))
        # Following code is for documentation illustration
        self.t1, self.t2 = t1, t2


if __name__ == "__main__":
    sol_mole = Molecule()
    sol_mole.construct_from_dat_file("input/h2o/STO-3G/geom.dat")
    sol_mole.obtain_mol_instance(basis="STO-3G")
    sol_mole.print_solution_05()
    # Following code is for documentation illustration
    import pickle
    d = {
        "mo_coeff": sol_mole.mo_coeff,
        "mo_energy": sol_mole.mo_energy,
        "t1": sol_mole.t1,
        "t2": sol_mole.t2,
    }
    with open("demo_data_h2o_sto3g.dat", "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
