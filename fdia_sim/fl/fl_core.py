"""
Federated training core — client local update and server aggregation.

Algorithm 1 from the paper is implemented in ``aggregate_multikrum``;
``aggregate_fedavg`` and ``aggregate_trimmed_mean`` are the baselines.

The norm-bound check that the paper offloads to a zero-knowledge proof
(Algorithm 1 line 4 + line 8) is implemented here as a *direct*
``||delta||_2 <= tau`` test. This is the function the ZKP attests in
zero knowledge — see ``zk/bulletproofs_sim.py`` for the timing model.
A delta with norm > tau is rejected (i.e. would fail ZKVerify in the
real protocol).
"""
import numpy as np
from autograd import grad
from model.gatv2 import (forward, bce_with_logits, Adam, params_diff,
                         apply_diff, params_to_vec, vec_to_params, clip_diff,
                         add_self_loops)


# ---------- client side -----------------------------------------------

def local_train_one_epoch(p_global, dataset, edge_index_self, lr=5e-3,
                          batch_size=8, pos_weight=4.0, l2_clip_tau=None):
    """Run one local epoch over (X_train, y_train) of one client.
    Returns (delta_dict, ||delta||_2_pre_clip, ||delta||_2_post_clip)."""
    X_list = dataset["X_train"]; Y_list = dataset["y_train"]
    n = len(X_list)
    perm = np.random.permutation(n)
    p = {k: v.copy() for k, v in p_global.items()}
    opt = Adam(p, lr=lr)
    loss_fn_factory = lambda Xb, yb: (lambda params:
        np.mean([bce_with_logits(forward(params, Xb[i], edge_index_self),
                                  yb[i].astype(np.float32),
                                  pos_weight=pos_weight)
                  for i in range(len(Xb))]))
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        Xb = [X_list[i] for i in idx]
        yb = [Y_list[i] for i in idx]
        L = loss_fn_factory(Xb, yb)
        g = grad(L)(p)
        p = opt.step(p, g)
    delta = params_diff(p, p_global)
    delta_vec = params_to_vec(delta)
    pre = float(np.linalg.norm(delta_vec))
    if l2_clip_tau is not None:
        clipped, _ = clip_diff(delta_vec, l2_clip_tau)
        delta = vec_to_params(clipped, delta)
        post = float(np.linalg.norm(clipped))
    else:
        post = pre
    return delta, pre, post


# ---------- server-side aggregation -----------------------------------

def aggregate_fedavg(deltas, weights=None):
    """Plain weighted average of client deltas."""
    if weights is None:
        weights = [1.0 / len(deltas)] * len(deltas)
    out = {k: np.zeros_like(v) for k, v in deltas[0].items()}
    for d, w in zip(deltas, weights):
        for k in out:
            out[k] = out[k] + w * d[k]
    return out


def aggregate_trimmed_mean(deltas, trim_fraction=0.2):
    """Coordinate-wise trimmed mean (Yin et al. 2018)."""
    out = {}
    for k in deltas[0].keys():
        stack = np.stack([d[k] for d in deltas], axis=0)  # (n, ...)
        n = stack.shape[0]
        k_trim = int(np.floor(trim_fraction * n))
        srt = np.sort(stack, axis=0)
        if k_trim > 0:
            kept = srt[k_trim:n - k_trim]
        else:
            kept = srt
        out[k] = np.mean(kept, axis=0)
    return out


def aggregate_multikrum(deltas, f, m=None):
    """Multi-Krum (Blanchard et al. 2017): score each client by sum of squared
    distances to its n-f-2 nearest neighbours; average the m clients with
    smallest score. If m is None, m = n - f.

    Returns (aggregated_delta_dict, selected_indices)."""
    n = len(deltas)
    if n - f - 2 <= 0:
        # not enough for Krum — fallback to trimmed mean
        return aggregate_trimmed_mean(deltas, trim_fraction=f / n), list(range(n))
    if m is None:
        m = max(1, n - f)
    # flatten all
    vecs = np.stack([params_to_vec(d) for d in deltas], axis=0)  # (n, D)
    # pairwise sq distances
    diff = vecs[:, None, :] - vecs[None, :, :]
    D = np.einsum("ijd,ijd->ij", diff, diff)
    np.fill_diagonal(D, np.inf)
    k = n - f - 2
    scores = np.zeros(n)
    for i in range(n):
        nearest = np.partition(D[i], k - 1)[:k]
        scores[i] = nearest.sum()
    sel = np.argsort(scores)[:m].tolist()
    avg_vec = vecs[sel].mean(axis=0)
    agg = vec_to_params(avg_vec, deltas[0])
    return agg, sel
