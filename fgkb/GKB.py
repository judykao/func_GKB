"""Functional Golub-Kahan bidiagonalization with thick restart and deflation.

The algorithm only sees four things:
    T        : f in domain   -> g in codomain
    T_adj    : g in codomain -> f in domain
    inner_dom, inner_cod : inner products of the two spaces
Norms are derived from the inner products.
"""


import numpy as np

def _norm(inner, u):
    return np.sqrt(np.real(inner(u, u)))


def check_orthogonality(func_list, inner):
    k = len(func_list)
    G = np.zeros((k, k), dtype=complex)
    for i in range(k):
        for j in range(k):
            G[i, j] = inner(func_list[i], func_list[j])
    return np.linalg.norm(G - np.eye(k), ord=2)


def deflated_operator(T, T_adj, U_known, inner_cod):
    """One-sided deflation: project the range of T away from U_known."""
    def apply(f):
        u = T(f)
        for u_k in U_known:
            u = u - inner_cod(u_k, u) * u_k
        return u
    return apply, T_adj


def _lanczos_bidiag(T, T_adj, inner_dom, inner_cod, v_init,
                    u_list=None, v_list=None, alpha=None, beta=None,
                    K=100, tau_tol=1e-12, check_orth_freq=50, verbose=False):
    u_list = [] if u_list is None else u_list
    v_list = [] if v_list is None else v_list
    alpha = [] if alpha is None else alpha
    beta = [] if beta is None else beta

    fail = False
    v_list.append(v_init.copy())
    j = len(v_list) - 1
    if verbose:
        print(f"Starting Lanczos bidiagonalization with j={j}, K={K}")

    while j < K:
        u = T(v_list[j]) if j == 0 else T(v_list[j]) - beta[-1] * u_list[-1]

        a = _norm(inner_cod, u)
        alpha.append(a)
        if a < 1e-14:
            if verbose:
                print(f"Rank deficient at step {j+1}")
            break

        u = u / a
        u_list.append(u.copy())

        v = T_adj(u) - a * v_list[j]
        for v_prev in v_list:                      # one-sided reorth (on V)
            v = v - inner_dom(v_prev, v) * v_prev

        b = _norm(inner_dom, v)
        beta.append(b)
        v = v / b
        v_list.append(v.copy())

        if verbose and (j + 1) % check_orth_freq == 0:
            tau_u = check_orthogonality(u_list, inner_cod)
            tau_v = check_orthogonality(v_list[:-1], inner_dom)
            tau = max(tau_u, tau_v)
            print(f"[Step {j+1}] α={a:.4e}, β={b:.4e}, orth: u={tau_u:.2e}, v={tau_v:.2e}")

            ### Check orthogonality and break if it fails and reclaim the last few vectors
            if tau > tau_tol:
                fail = True
                print(f"Orthogonality check failed at step {j+1} with tau={tau:.4e}")
                final_beta = beta[-(check_orth_freq + 1)] if len(beta) > check_orth_freq else 0
                return alpha, beta[:-(check_orth_freq + 1)], u_list, v_list[:-(check_orth_freq + 1)], final_beta, v_list[-1], fail

        if b < 1e-14:
            fail = True
            if verbose:
                print(f"Breakdown at step {j+1}")
            break
        j += 1

    final_beta = beta[-1] if beta else 0
    return alpha, beta[:-1], u_list, v_list[:-1], final_beta, v_list[-1], fail


def golub_kahan_thick_restart(T, T_adj, inner_dom, inner_cod, f0,
                              K=200, n_threshold=400, restart_threshold=0.5,
                              max_restarts=10, res_tol=1e-16, tau_tol=1e-7,
                              check_orth_freq=50, verbose=False):
    T0, T_adj0 = T, T_adj                           # undeflated operator
    U, V, S = [], [], []
    v = f0 / _norm(inner_dom, f0)

    deflation_iter = 0
    while len(S) < n_threshold:
        if deflation_iter > 0:
            T, T_adj = deflated_operator(T0, T_adj0, U, inner_cod)
            v = b_vec
            for v_prev in V:
                v = v - inner_dom(v_prev, v) * v_prev
            v = v / _norm(inner_dom, v)

        u_list, v_list = [], []
        alpha, beta = [], []
        l = 0
        for restart_iter in range(max_restarts):
            if verbose:
                print(f"\n{'='*60}\nDeflation iteration {deflation_iter}, "
                      f"Restart iteration {restart_iter}\n{'='*60}")

            alpha, beta, u_list, v_list, b, b_vec, fail = _lanczos_bidiag(
                T, T_adj, inner_dom, inner_cod, v, u_list, v_list, alpha, beta,
                K, tau_tol=tau_tol, check_orth_freq=check_orth_freq, verbose=verbose)

            k = len(v_list)
            B = np.zeros((k, k), dtype=complex)
            if l == 0:
                for j in range(k):
                    B[j, j] = alpha[j]
                    if j < k - 1:
                        B[j, j + 1] = beta[j]
            else:
                idx = 0
                for j in range(k):
                    B[j, j] = alpha[j]
                    if l <= j < k - 1:
                        B[j, j + 1] = beta[idx]
                        idx += 1
                for j in range(l):
                    B[j, l] = rho[j]

            U_b, s_b, Vh_b = np.linalg.svd(B, full_matrices=False)
            V_b = Vh_b.conj().T

            conv_n, flag = 0, False
            for i in range(len(s_b)):
                residual = b * abs(U_b[-1, i])
                if verbose:
                    print(f"σ_{i+1}: {s_b[i]:.6e}, residual: {residual:.6e}", end="")
                if residual < res_tol:
                    if not flag:
                        conv_n += 1
                    if verbose:
                        print("  → conv!")
                else:
                    flag = True
                    if verbose:
                        print()

            l = conv_n + (k - conv_n) // 2
            if verbose:
                print(f"\nConverged in this iteration: {conv_n}")

            if conv_n > np.round(K * restart_threshold):
                U += [sum(u_list[j] * U_b[j, i] for j in range(k)) for i in range(conv_n)]
                V += [sum(v_list[j] * V_b[j, i] for j in range(k)) for i in range(conv_n)]
                S += s_b[:conv_n].tolist()
                print(f"\nTotal converged: {len(S)}")
                break

            if len(S) >= n_threshold or fail:
                return U, V, S

            new_u_list, new_v_list, alpha = [], [], []
            for i in range(l):
                new_u_list.append(sum(u_list[j] * U_b[j, i] for j in range(k)))
                new_v_list.append(sum(v_list[j] * V_b[j, i] for j in range(k)))
                alpha.append(s_b[i])
            new_v_list.append(b_vec)

            u = T(b_vec)
            rho = []
            for i, u_prev in enumerate(new_u_list):
                rho_i = b * U_b[-1, i]
                rho.append(rho_i)
                u = u - rho_i * u_prev

            a_l1 = _norm(inner_cod, u)
            u = u / a_l1
            new_u_list.append(u)
            alpha.append(a_l1)

            u_list, v_list = new_u_list.copy(), new_v_list.copy()

            tau_u = check_orthogonality(u_list, inner_cod)
            tau_v = check_orthogonality(v_list, inner_dom)
            print(f"[Deflation {deflation_iter}] Restart {restart_iter+1}: "
                  f"τ_u={tau_u:.2e}, τ_v={tau_v:.2e}, max τ={max(tau_u, tau_v):.2e}")

            v = T_adj(u) - a_l1 * b_vec
            for v_prev in v_list:
                v = v - inner_dom(v_prev, v) * v_prev
            b_l1 = _norm(inner_dom, v)
            v = v / b_l1
            beta = [b_l1]

        deflation_iter += 1
    return U, V, S