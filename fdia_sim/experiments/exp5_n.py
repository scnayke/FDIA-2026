"""
Experiment 5 — Sensitivity to number of clients n (Sec. VIII.H of paper).

Sweep n in {4, 5, 8, 10} at fixed total training set size of ~400 snapshots
on case33bw, ours (MK-GATv2 + ZKP), f=0.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.runner import run_federated

N_VALUES = [4, 5, 8, 10]
ROUNDS = 20
TOTAL_TR = 400
TOTAL_TE = 125
SEEDS = [42, 1234]

def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "results", "exp5_n.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        done = set((r["n"], r["seed"]) for r in out["runs"])
    else:
        out = {"config": {"rounds": ROUNDS, "total_tr": TOTAL_TR}, "runs": []}
        done = set()
    t0 = time.perf_counter()
    for n in N_VALUES:
        for seed in SEEDS:
            if (n, seed) in done:
                continue
            n_tr = TOTAL_TR // n
            n_te = TOTAL_TE // n
            ts = time.perf_counter()
            print(f"[n={n} seed={seed}] running...", flush=True)
            res = run_federated(
                case="case33bw", model_type="gatv2",
                num_utilities=n,
                n_train_per=n_tr, n_test_per=n_te,
                rounds=ROUNDS, seed=seed,
                aggregator="multikrum", f_byzantine=0, byz_attack="honest",
                use_zk_bound=True, tau=8.0,
                hidden=8, heads=4, lr=1e-2, pos_weight=8.0,
                eval_every=ROUNDS, verbose=False,
            )
            m = res["final_metric"]
            print(f"  -> F1={m['f1']:.3f} auc={m['auc']:.3f}  "
                  f"({time.perf_counter()-ts:.1f}s)", flush=True)
            out["runs"].append({"n": n, "seed": seed, "n_tr": n_tr,
                                "final_metric": m,
                                "wall_s": time.perf_counter() - ts})
            done.add((n, seed))
            with open(out_path, "w") as fh:
                json.dump(out, fh, indent=1, default=float)
    print(f"\nDone. Total: {time.perf_counter()-t0:.1f}s")

if __name__ == "__main__":
    main()
