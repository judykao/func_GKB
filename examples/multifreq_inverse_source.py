"""multifreq inverse source operator: L^2(B_1) -> L^2([0,k_max] x S^2)."""
import numpy as np
from fgkb.sampling import gen_quasi_mc_points

PARAMS = dict(
    k_max=20, 
    N_kappa=100, 
    N_ball=1000, 
    N_x_hat=100
)

SOLVER = dict(
    K=400, 
    n_threshold=1200, 
    restart_threshold=0.5,
    max_restarts=10, 
    res_tol=1e-13, 
    tau_tol=1e-7
)


def build(k_max, N_kappa, N_ball, N_x_hat):
    ball = gen_quasi_mc_points(3, N_ball, [-1, -1, -1], [1, 1, 1])
    w_B = 8.0 / N_ball

    kappa = np.linspace(0, k_max, N_kappa)
    w_kappa = k_max / N_kappa

    x_hat = gen_quasi_mc_points(2, N_x_hat, [0, 0], [np.pi, 2 * np.pi])
    w_x_hat = 4 * np.pi / N_x_hat

    sp, cp = np.sin(x_hat[:, 0])[None, :], np.cos(x_hat[:, 0])[None, :]
    st, ct = np.sin(x_hat[:, 1])[None, :], np.cos(x_hat[:, 1])[None, :]
    b1, b2, b3 = (ball[:, i][:, None] for i in range(3))

    rows = [np.exp(1j * kap * (b1 * sp * ct + b2 * sp * st + b3 * cp)).T for kap in kappa]
    E = np.vstack(rows)                       # (N_kappa*N_x_hat, N_ball)
    print(f"E.shape: {E.shape}")

    return dict(
        T=lambda f: E.conj() @ (f * w_B),
        T_adj=lambda g: E.T @ (g * w_kappa * w_x_hat),
        inner_dom=lambda u, v: np.sum(np.conjugate(u) * v) * w_B,
        inner_cod=lambda u, v: np.sum(np.conjugate(u) * v) * w_kappa * w_x_hat,
        f0=np.ones(N_ball, dtype=complex),
    )