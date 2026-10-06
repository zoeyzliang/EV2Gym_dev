"""
powerflow.py
============
Batched fixed-point ("tensor") power flow for radial feeders with
constant-power loads, in plain NumPy.

This is the same algorithm and network model as EV2Gym's GridTensor
(ev2gym/models/grid_utility/grid_tensor.py, from RL-ADN) in its
constant-power "tensor" mode:

    Ybus from the line data (R, X in ohm on the feeder base, B, TAP)
    K = −Ydd⁻¹,  L = K·Yds   (slack bus = first node, V_slack = 1 pu)
    v ← K · conj(S / v) + L,  S = (P + jQ) / s_base,  flat start

It is re-implemented here so the feeder environment does not depend on
EV2Gym's grid utilities, which import numba, psutil and pandapower at module
load (numba is absent on the M3 environment; pandapower 2.13 does not import
under numpy 2). Equivalence with GridTensor is checked in
tests/test_feeder_env.py.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class TensorPowerFlow:
    """
    Parameters
    ----------
    nodes, lines : pd.DataFrame
        EV2Gym network files (Nodes_*.csv: NODES, Tb, PD, QD, ...;
        Lines_*.csv: FROM, TO, R, X, B, STATUS, TAP).
    s_base : float   kVA (EV2Gym default 1000)
    v_base : float   kV  (EV2Gym default 11)
    """

    def __init__(self, nodes: pd.DataFrame, lines: pd.DataFrame,
                 s_base: float = 1000.0, v_base: float = 11.0):
        self.s_base, self.v_base = s_base, v_base
        nb, nl = len(nodes), len(lines)
        slack = nodes.index[nodes["Tb"] == 1][0]
        if slack != 0:
            raise ValueError("slack bus must be the first node (as in GridTensor)")

        z_base = v_base ** 2 * 1000.0 / s_base
        stat = lines.iloc[:, 5].to_numpy(float)
        r, x = lines.iloc[:, 2].to_numpy(float), lines.iloc[:, 3].to_numpy(float)
        Ys = stat / ((r + 1j * x) / z_base)
        Bc = stat * lines.iloc[:, 4].to_numpy(float) * z_base
        tap = stat * lines.iloc[:, 6].to_numpy(float)
        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / tap
        Yft = -Ys / tap
        Ytf = Yft
        f = lines.iloc[:, 0].to_numpy(int) - 1
        t = lines.iloc[:, 1].to_numpy(int) - 1
        Cf = csr_matrix((np.ones(nl), (np.arange(nl), f)), (nl, nb))
        Ct = csr_matrix((np.ones(nl), (np.arange(nl), t)), (nl, nb))
        i = np.r_[np.arange(nl), np.arange(nl)]
        Yf = csr_matrix((np.r_[Yff, Yft], (i, np.r_[f, t])), (nl, nb))
        Yt = csr_matrix((np.r_[Ytf, Ytt], (i, np.r_[f, t])), (nl, nb))
        Ybus = (Cf.T @ Yf + Ct.T @ Yt).toarray()

        Ydd = Ybus[1:, 1:]
        Yds = Ybus[1:, 0]
        self.K = -np.linalg.inv(Ydd)            # (nb-1, nb-1)
        self.L = self.K @ Yds                   # (nb-1,)
        self.KT = self.K.T.copy()
        self.n = nb - 1

    def solve(self, P: np.ndarray, Q: np.ndarray, tol: float = 1e-9, max_iter: int = 300):
        """
        Complex bus voltages (pu, slack excluded) for scenarios P, Q of shape
        (S, nb-1) in kW / kvar (loads positive). Returns (V, converged).
        """
        P = np.atleast_2d(np.asarray(P, float)); Q = np.atleast_2d(np.asarray(Q, float))
        S = (P + 1j * Q) / self.s_base
        Sc = np.conj(S)
        v = np.ones(S.shape, dtype=complex)
        for _ in range(max_iter):
            v_new = (Sc / np.conj(v)) @ self.KT + self.L
            done = np.max(np.abs(np.abs(v_new) - np.abs(v))) < tol
            v = v_new
            if done:
                return v, True
        return v, False
