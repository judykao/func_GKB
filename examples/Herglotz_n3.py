"""Herglotz wave operator (n = 3, single wavenumber): L^2(S^2) -> L^2(B_1)."""
import numpy as np
from fgkb.sampling import gen_quasi_mc_points

PARAMS = dict(
    kappa=5, 
    N_theta=1000, 
    N_x=10000
)

SOLVER = dict(
    K=100, 
    n_threshold=200, 
    restart_threshold=0.5,
    max_restarts=10, 
    res_tol=1e-13, 
    tau_tol=1e-7
)


def build(kappa, N_theta, N_x):
    theta = gen_quasi_mc_points(2, N_theta, [0, 0], [np.pi, 2 * np.pi])
    w_theta = 4 * np.pi / N_theta
    print(f"theta.shape: {theta.shape}")
    
    x = gen_quasi_mc_points(3, N_x, [-1, -1, -1], [1, 1, 1])
    w_x = 8.0 / N_x
    print(f"x.shape: {x.shape}")

    cos_phi = np.cos(theta[:, 0])[None, :]
    sin_phi = np.sin(theta[:, 0])[None, :]
    cos_th = np.cos(theta[:, 1])[None, :]
    sin_th = np.sin(theta[:, 1])[None, :]
    x1, x2, x3 = (x[:, i][:, None] for i in range(3))

    E = np.exp(1j * kappa * (x1 * (sin_phi * cos_th) + x2 * (sin_phi * sin_th) + x3 * cos_phi))
    print(f"E.shape: {E.shape}")                     # (N_x, N_theta)

    return dict(
        T=lambda f: kappa * E.dot(f * w_theta),
        T_adj=lambda g: kappa * E.conj().T.dot(g * w_x),
        inner_dom=lambda u, v: np.sum(np.conjugate(u) * v) * w_theta,
        inner_cod=lambda u, v: np.sum(np.conjugate(u) * v) * w_x,
        f0=np.ones(N_theta, dtype=complex),
    )