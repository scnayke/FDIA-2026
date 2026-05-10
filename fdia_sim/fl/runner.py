"""
Top-level federated training driver.

Implements the per-round protocol from Algorithm 1 of the paper:
  1. Server broadcasts w_t.
  2. Each honest client runs one local epoch and clips its delta to ||·||_2 <= tau.
     The ZK norm-bound proof is *modelled* (via zk/bulletproofs_sim) — for the
     correctness of the protocol we use the predicate ||delta||_2 <= tau directly.
  3. Byzantine clients submit malicious deltas (possibly violating the bound).
  4. Server runs ZKVerify (= bound check); rejects clients whose deltas
     fail. Then aggregates by FedAvg / Multi-Krum / Trimmed Mean.
  5. Server appends a hash-chain record.
"""
import time
import numpy as np
from autograd import grad

from grid.data import make_federated_dataset
from model.gatv2 import (init_params as init_gatv2, num_params, add_self_loops,
                          forward as forward_gatv2,
                          bce_with_logits, params_to_vec, vec_to_params,
                          params_diff, apply_diff, clip_diff, Adam)
from model.mlp import (init_mlp_per_bus, forward_mlp_per_bus,
                       init_mlp_flat,    forward_mlp_flat)
from fl.fl_core import (aggregate_fedavg, aggregate_multikrum,
                         aggregate_trimmed_mean)
from fl.evaluation import evaluate_combined
from attacks.byz import ATTACK_FNS, attack_label_flip_apply
from zk.bulletproofs_sim import (estimated_prove_time_s, estimated_verify_time_s,
                                  proof_size_bytes, simulate_proof_bytes,
                                  commitment_check)
from audit.hashchain import HashChain, hash_vec


# ------------- model dispatch -----------------------------------------

def make_model(model_type, F_in, N_nodes, hidden, heads, rng):
    if model_type == "gatv2":
        return init_gatv2(F_in=F_in, hidden=hidden, heads=heads, rng=rng), forward_gatv2
    elif model_type == "mlp_perbus":
        return init_mlp_per_bus(F_in, hidden, rng=rng), forward_mlp_per_bus
    elif model_type == "mlp_flat":
        return init_mlp_flat(F_in, N_nodes, hidden, rng=rng), forward_mlp_flat
    else:
        raise ValueError(model_type)


def local_train(p_global, dataset, edge_index_self, forward_fn,
                lr=5e-3, batch_size=8, pos_weight=4.0, l2_clip_tau=None,
                rng=None):
    """Run one local epoch, return (delta_dict, npre, npost)."""
    X_list = dataset["X_train"]; Y_list = dataset["y_train"]
    n = len(X_list)
    rng = rng or np.random.default_rng()
    perm = rng.permutation(n)
    p = {k: v.copy() for k, v in p_global.items()}
    opt = Adam(p, lr=lr)
    import autograd.numpy as anp
    def L(params, Xb, yb):
        losses = []
        for X, y in zip(Xb, yb):
            z = forward_fn(params, X, edge_index_self)
            losses.append(bce_with_logits(z, y.astype(np.float32),
                                          pos_weight=pos_weight))
        return anp.mean(anp.array(losses))
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        Xb = [X_list[i] for i in idx]
        yb = [Y_list[i] for i in idx]
        loss_p = lambda params: L(params, Xb, yb)
        g = grad(loss_p)(p)
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


def run_federated(
    case: str = "case33bw",
    model_type: str = "gatv2",           # gatv2 | mlp_perbus | mlp_flat
    num_utilities: int = 5,
    n_train_per: int = 80,
    n_test_per: int = 40,
    rounds: int = 50,
    seed: int = 42,
    aggregator: str = "multikrum",      # fedavg | multikrum | trimmed_mean
    f_byzantine: int = 0,               # number of Byzantine clients
    byz_attack: str = "honest",         # signflip | gauss | unbounded | labelflip
    use_zk_bound: bool = True,
    tau: float = 5.0,                    # ||delta||_2 norm bound
    hidden: int = 8,
    heads: int = 4,
    lr: float = 1e-2,
    pos_weight: float = 8.0,
    eval_every: int = 10,
    verbose: bool = True,
    cached_datasets=None,                # if provided, skip dataset build
):
    """Run one federated training experiment. Returns a results dict."""
    t0 = time.perf_counter()
    # ---- Build per-utility data ----
    datasets = make_federated_dataset(case, num_utilities, n_train_per, n_test_per, seed)
    F_in = datasets[0]["X_train"][0].shape[1]
    N_nodes = datasets[0]["X_train"][0].shape[0]
    ei = datasets[0]["edge_index"]
    ei_sl = add_self_loops(ei, N_nodes)

    # ---- Init model ----
    rng = np.random.default_rng(seed)
    p_global, forward_fn = make_model(model_type, F_in, N_nodes, hidden, heads, rng)
    D = num_params(p_global)

    # ---- Pick Byzantine clients ----
    byz_rng = np.random.default_rng(seed + 7)
    byz_ids = set(byz_rng.choice(num_utilities, size=f_byzantine, replace=False).tolist()) \
              if f_byzantine > 0 else set()

    history = {"rounds": [], "metrics": [], "rejected_per_round": [],
               "selected_per_round": [], "delta_norms_pre": [],
               "delta_norms_post": []}
    chain = HashChain()
    cumulative = {"local_train_s": 0.0, "agg_s": 0.0,
                  "zk_prove_s_sim": 0.0, "zk_verify_s_sim": 0.0,
                  "anchor_s": 0.0}

    t_setup = time.perf_counter() - t0

    for r in range(rounds):
        rs = time.perf_counter()

        # ---- Per-client local training ----
        deltas = []
        norms_pre = []; norms_post = []
        train_s = 0.0
        for u, d in enumerate(datasets):
            # If labelflip Byzantine: mutate labels before training
            d_u = d
            if u in byz_ids and byz_attack == "labelflip":
                d_u = dict(d); d_u["y_train"] = attack_label_flip_apply(d["y_train"])
            t_loc = time.perf_counter()
            try:
                delta, npre, npost = local_train(
                    p_global, d_u, ei_sl, forward_fn=forward_fn,
                    lr=lr, pos_weight=pos_weight,
                    l2_clip_tau=tau if (u not in byz_ids) else None,
                    rng=np.random.default_rng(seed * 10000 + r * 100 + u),
                )
            except Exception as e:
                # if a client diverges, treat as a no-op contribution
                delta = {k: np.zeros_like(v) for k, v in p_global.items()}
                npre = npost = 0.0
            train_s += time.perf_counter() - t_loc
            # If Byzantine and not labelflip: apply post-hoc gradient corruption
            if u in byz_ids and byz_attack in ("signflip", "gauss", "unbounded"):
                delta = ATTACK_FNS[byz_attack](delta, byz_rng)
                npost = float(np.linalg.norm(params_to_vec(delta)))
            deltas.append(delta)
            norms_pre.append(npre); norms_post.append(npost)
        cumulative["local_train_s"] += train_s

        # ---- Simulated ZK prove + verify (timing only) ----
        if use_zk_bound:
            zk_p = sum(estimated_prove_time_s(D) for _ in deltas)
            zk_v = sum(estimated_verify_time_s(D) for _ in deltas)
            cumulative["zk_prove_s_sim"]  += zk_p
            cumulative["zk_verify_s_sim"] += zk_v

        # ---- Server-side ZKVerify == bound check ----
        rejected_ids = []
        kept_idx = list(range(num_utilities))
        if use_zk_bound:
            kept_idx = []
            for i, d_i in enumerate(deltas):
                if commitment_check(params_to_vec(d_i), tau):
                    kept_idx.append(i)
                else:
                    rejected_ids.append(i)
        deltas_kept = [deltas[i] for i in kept_idx]

        # ---- Aggregate ----
        ts_a = time.perf_counter()
        if len(deltas_kept) == 0:
            agg = {k: np.zeros_like(v) for k, v in p_global.items()}
            sel = []
        elif aggregator == "fedavg":
            agg = aggregate_fedavg(deltas_kept)
            sel = list(range(len(deltas_kept)))
        elif aggregator == "trimmed_mean":
            agg = aggregate_trimmed_mean(deltas_kept, trim_fraction=0.2)
            sel = list(range(len(deltas_kept)))
        elif aggregator == "multikrum":
            agg, sel = aggregate_multikrum(deltas_kept,
                                            f=max(1, f_byzantine - len(rejected_ids)))
        else:
            raise ValueError(aggregator)
        cumulative["agg_s"] += time.perf_counter() - ts_a

        # ---- Apply update ----
        p_global = apply_diff(p_global, agg)

        # ---- Anchor ----
        ts_an = time.perf_counter()
        accepted_orig = [kept_idx[i] for i in sel]
        accepted_hashes = [hash_vec(params_to_vec(deltas[i]).tobytes()) for i in accepted_orig]
        model_hash = hash_vec(params_to_vec(p_global).tobytes())
        chain.append(r, model_hash, accepted_hashes, rejected_ids)
        cumulative["anchor_s"] += time.perf_counter() - ts_an

        # ---- Track ----
        history["rejected_per_round"].append(rejected_ids)
        history["selected_per_round"].append(accepted_orig)
        history["delta_norms_pre"].append(norms_pre)
        history["delta_norms_post"].append(norms_post)

        # ---- Eval ----
        if (r + 1) % eval_every == 0 or r == rounds - 1:
            m = evaluate_combined(p_global, datasets, ei_sl, forward_fn)
            history["rounds"].append(r + 1)
            history["metrics"].append(m)
            if verbose:
                print(f"  [r={r+1:3d}] f1={m['f1']:.3f} prec={m['precision']:.3f} "
                      f"rec={m['recall']:.3f} auc={m['auc']:.3f} "
                      f"sel={accepted_orig} rej={rejected_ids} "
                      f"round_s={time.perf_counter()-rs:.1f}")

    return {
        "case": case,
        "config": {"num_utilities": num_utilities, "rounds": rounds, "seed": seed,
                   "aggregator": aggregator, "f_byzantine": f_byzantine,
                   "byz_attack": byz_attack, "use_zk_bound": use_zk_bound,
                   "tau": tau, "hidden": hidden, "heads": heads,
                   "n_train_per": n_train_per, "n_test_per": n_test_per},
        "num_params": D,
        "final_metric": history["metrics"][-1] if history["metrics"] else None,
        "history": history,
        "byz_ids": sorted(list(byz_ids)),
        "cumulative_s": cumulative,
        "proof_size_bytes": proof_size_bytes(D),
        "comm_bytes_per_client_per_round": int(D * 4 + proof_size_bytes(D) + 32),
        "setup_s": t_setup,
    }
