"""
Experiment 3 — Per-round latency breakdown (Sec. VIII-E of paper).

We instrument run_federated and average over 20 rounds with n=10 clients
on case33bw. Reports actual measured wall-clock timings for:

  - local_train_s     : per-round local epoch wall time (sum across clients)
  - zk_prove_s_sim    : Bulletproofs prove time (calibrated model)
  - zk_verify_s_sim   : Bulletproofs verify time (calibrated model)
  - agg_s             : Multi-Krum aggregation
  - anchor_s          : hash-chain append

Plus a separate detection-path (inference-only) measurement.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fl.runner import run_federated, make_model
from grid.data import make_federated_dataset
from grid.grid_sim import edge_index, load_net
from model.gatv2 import add_self_loops, num_params, forward as forward_gatv2

ROUNDS = 20
N_UTILS = 10
N_TRAIN = 100
N_TEST  = 30

def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "results", "exp3_latency.json")
    print(f"Running {ROUNDS}-round latency study on case33bw, n={N_UTILS}...")
    res = run_federated(
        case="case33bw", model_type="gatv2",
        num_utilities=N_UTILS,
        n_train_per=N_TRAIN, n_test_per=N_TEST,
        rounds=ROUNDS, seed=42,
        aggregator="multikrum", f_byzantine=0,
        byz_attack="honest",
        use_zk_bound=True, tau=8.0,
        hidden=8, heads=4, lr=1e-2, pos_weight=8.0,
        eval_every=ROUNDS, verbose=False,
    )
    cum = res["cumulative_s"]
    R = ROUNDS
    per_round_avg = {k: v / R for k, v in cum.items()}
    print(f"\nPer-round averaged costs (s):")
    for k, v in per_round_avg.items():
        print(f"  {k:<20s}: {v*1000:8.1f} ms")
    print(f"\nProof size: {res['proof_size_bytes']} bytes; comm/round/client: {res['comm_bytes_per_client_per_round']} bytes")
    print(f"num params: {res['num_params']}")

    # ---- Detection-path latency (inference only, not blocked by FL/ZKP) ----
    # Run forward N times on a fresh snapshot, measure mean
    rng = np.random.default_rng(0)
    p, fwd = make_model("gatv2", 6, 33, 8, 4, rng)
    # use one utility's snapshot as input
    ds = make_federated_dataset("case33bw", 1, 10, 1, seed=42)
    X = ds[0]["X_train"][0]
    ei = ds[0]["edge_index"]
    ei_sl = add_self_loops(ei, X.shape[0])
    # warm-up
    for _ in range(5):
        _ = fwd(p, X, ei_sl)
    K = 50
    t0 = time.perf_counter()
    for _ in range(K):
        _ = fwd(p, X, ei_sl)
    dt33 = (time.perf_counter() - t0) / K * 1000.0  # ms
    print(f"\nDetection inference (case33bw): {dt33:.2f} ms")

    # case118
    ds2 = make_federated_dataset("case118", 1, 10, 1, seed=42)
    X2 = ds2[0]["X_train"][0]
    ei2 = ds2[0]["edge_index"]
    ei2_sl = add_self_loops(ei2, X2.shape[0])
    rng2 = np.random.default_rng(0)
    p2, fwd2 = make_model("gatv2", 6, X2.shape[0], 8, 4, rng2)
    for _ in range(3):
        _ = fwd2(p2, X2, ei2_sl)
    K = 25
    t0 = time.perf_counter()
    for _ in range(K):
        _ = fwd2(p2, X2, ei2_sl)
    dt118 = (time.perf_counter() - t0) / K * 1000.0
    print(f"Detection inference (case118):  {dt118:.2f} ms")

    out = {
        "rounds": R,
        "num_utilities": N_UTILS,
        "num_params": res["num_params"],
        "proof_size_bytes": res["proof_size_bytes"],
        "comm_bytes_per_client_per_round": res["comm_bytes_per_client_per_round"],
        "per_round_avg_ms": {k: v * 1000.0 for k, v in per_round_avg.items()},
        "detection_inference_ms": {"case33bw": dt33, "case118": dt118},
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
