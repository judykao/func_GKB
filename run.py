"""Usage:
    python run.py examples/multifreq_inverse_source.py
    python run.py examples/multifreq_inverse_source.py --out my_run --verbose

An example file defines
    build(**PARAMS) -> dict(T, T_adj, inner_dom, inner_cod, f0)
    PARAMS  (dict, optional)  problem parameters
    SOLVER  (dict, optional)  solver parameters
All settings live in the example file; edit them there.

Each run writes a new folder results/<example>_<time>/ (or results/<out>/):
    solution.npz  (U, S, V; rows are functions)   S.png   params.json
"""
import argparse
import datetime
import importlib.util
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fgkb.GKB import golub_kahan_thick_restart


def load(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("example")
    p.add_argument("--out", default=None, help="result folder name under results/")
    p.add_argument("--verbose", action="store_true", help="print solver progress")
    args = p.parse_args()

    mod = load(args.example)
    params = dict(getattr(mod, "PARAMS", {}))
    solver = dict(getattr(mod, "SOLVER", {}))
    solver["verbose"] = args.verbose
    stem = pathlib.Path(args.example).stem
    name = args.out or f"{stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    outdir = pathlib.Path("results") / name
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "params.json").write_text(
        json.dumps(dict(example=args.example, PARAMS=params, SOLVER=solver), indent=2))

    prob = mod.build(**params)
    U, V, S = golub_kahan_thick_restart(
        prob["T"], prob["T_adj"], prob["inner_dom"], prob["inner_cod"],
        prob["f0"], **solver)

    S = np.asarray(S)
    np.savez(outdir / "solution.npz", U=np.array(U), S=S, V=np.array(V))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, len(S) + 1), S, ".-", ms=3)
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.set_title(stem)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "S.png", dpi=150)
    print(f"saved {len(S)} singular values -> {outdir}/")


if __name__ == "__main__":
    main()