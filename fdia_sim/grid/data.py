"""
Per-utility data generator for federated FDIA detection.

Each utility owns the SAME grid topology but a private set of operating
snapshots produced by perturbing the base load profile within a
utility-specific range; FDIA samples are crafted with a utility-specific
sparsity/magnitude distribution. This is the canonical horizontal FL
setup with non-IID local distributions.

A "sample" is a tuple (X, y) where X is the per-bus 6-dim feature
matrix (after standardisation w.r.t. the global stats fitted on
clean snapshots) and y is the per-bus binary label
(1 = bus measurement compromised, 0 = clean).

Snapshots are reproducible via a per-utility RNG seed.
"""
import numpy as np
from grid.grid_sim import (load_net, run_one, per_bus_features,
                            measurement_jacobian, craft_fdia,
                            label_compromised_buses, edge_index, standardize)


def make_utility_dataset(
    case: str,
    util_id: int,
    n_train: int,
    n_test: int,
    seed: int,
    *,
    load_low=0.7, load_high=1.3,
    sparsity_lo=3, sparsity_hi=6,
    mag_lo=0.10, mag_hi=0.30,
    attack_prob=0.5,
    q_coupling=0.5,
):
    """Generate (Xtr, Ytr, Xte, Yte, ei) for a single utility."""
    rng = np.random.default_rng(seed * 1000 + util_id)
    net = load_net(case)
    ei = edge_index(net)
    H = measurement_jacobian(net, run_one(net, scale=1.0))

    # First, compute global standardisation stats from a small clean sweep
    clean_feats = []
    for _ in range(64):
        sc = rng.uniform(load_low, load_high)
        try:
            s = run_one(net, scale=sc)
            clean_feats.append(per_bus_features(s, net))
        except Exception:
            continue
    pool = np.concatenate(clean_feats, axis=0)
    _, stats = standardize(pool, stats=None)

    def gen(n_samples):
        X_list, Y_list = [], []
        attempts = 0
        while len(X_list) < n_samples and attempts < n_samples * 4:
            attempts += 1
            sc = rng.uniform(load_low, load_high)
            try:
                s = run_one(net, scale=sc)
            except Exception:
                continue
            feats = per_bus_features(s, net)
            attacked = rng.uniform() < attack_prob
            if attacked:
                k = int(rng.integers(sparsity_lo, sparsity_hi + 1))
                mag = rng.uniform(mag_lo, mag_hi)
                a, c = craft_fdia(H, sparsity=k, magnitude=mag, rng=rng)
                # The unobservable FDIA biases the *measurement* vector. We
                # express the effect on per-bus features by adjusting the
                # active-power injection P_v (column 2) since H represents
                # power-flow sensitivities in the DC approximation, and
                # also add a coupled bias to Q_v (column 3) to model the
                # AC residual leakage that any practical FDIA produces.
                feats[:, 2] = feats[:, 2] + a.astype(np.float32)
                feats[:, 3] = feats[:, 3] + (q_coupling * a).astype(np.float32)
                y = label_compromised_buses(c)
            else:
                y = np.zeros(feats.shape[0], dtype=np.int64)
            feats_z, _ = standardize(feats, stats=stats)
            X_list.append(feats_z)
            Y_list.append(y)
        if len(X_list) < n_samples:
            # If pp.runpp diverged a lot, pad with last successful one
            while len(X_list) < n_samples:
                X_list.append(X_list[-1].copy()); Y_list.append(Y_list[-1].copy())
        return X_list, Y_list

    Xtr, Ytr = gen(n_train)
    Xte, Yte = gen(n_test)
    return {"X_train": Xtr, "y_train": Ytr,
            "X_test":  Xte, "y_test":  Yte,
            "edge_index": ei, "stats": stats, "H": H}


def make_federated_dataset(case: str, num_utilities: int, n_train_per: int,
                            n_test_per: int, seed: int = 42, *,
                            heterogeneous: bool = True):
    """Build per-utility datasets. If heterogeneous=True, vary load and
    attack ranges across utilities to induce non-IID."""
    datasets = []
    for u in range(num_utilities):
        if heterogeneous:
            # spread utilities across different load/attack regimes
            lo = 0.6 + 0.04 * u
            hi = lo + 0.4
            sp_lo = 3 + (u % 2)
            sp_hi = sp_lo + 3
            mag_lo = 0.08 + 0.02 * (u % 3)
            mag_hi = mag_lo + 0.20
        else:
            lo, hi = 0.7, 1.3
            sp_lo, sp_hi = 3, 6
            mag_lo, mag_hi = 0.10, 0.30
        d = make_utility_dataset(
            case, u, n_train_per, n_test_per, seed,
            load_low=lo, load_high=hi,
            sparsity_lo=sp_lo, sparsity_hi=sp_hi,
            mag_lo=mag_lo, mag_hi=mag_hi,
        )
        datasets.append(d)
    return datasets
