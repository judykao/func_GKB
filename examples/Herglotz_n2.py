"""Herglotz wave operator (n = 2, single wavenumber): L^2(S^1) -> L^2(B_1)."""
import numpy as np
from fgkb.sampling import gen_quasi_mc_points

PARAMS = dict(
    kappa=30, 
    N_theta=360, 
    N_x=10000
)

SOLVER = dict(
    K=150, 
    n_threshold=100, 
    restart_threshold=0.5,
    max_restarts=10, 
    res_tol=1e-10, 
    tau_tol=1e-7
)


def build(kappa, N_theta, N_x):
    theta = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)
    w_theta = 2 * np.pi / N_theta
    print(f"theta.shape: {theta.shape}")

    x = gen_quasi_mc_points(2, N_x, [-1, -1], [1, 1])
    w_x = 4.0 / N_x
    print(f"x.shape: {x.shape}")

    cos_th = np.cos(theta)[None, :]
    sin_th = np.sin(theta)[None, :]
    x1, x2 = (x[:, i][:, None] for i in range(2))

    E = np.exp(1j * kappa * (x1 * cos_th + x2 * sin_th))
    print(f"E.shape: {E.shape}")                     # (N_x, N_theta)

    return dict(
        T=lambda f: np.sqrt(kappa) * E.dot(f * w_theta),
        T_adj=lambda g: np.sqrt(kappa) * E.conj().T.dot(g * w_x),
        inner_dom=lambda u, v: np.sum(np.conjugate(u) * v) * w_theta,
        inner_cod=lambda u, v: np.sum(np.conjugate(u) * v) * w_x,
        f0=np.ones(N_theta, dtype=complex),
    )