"""
Experiment 1 — Main detection performance (Sec. VIII-C of paper).

Six methods x two cases x two seeds at f=0.
Saves a JSON to results/exp1_main.json with all per-run final metrics.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.runner import run_federated

CASES        = ["case33bw", "case118"]
SEEDS        = [42, 1234]
ROUNDS       = 30
EVAL_EVERY   = 10
N_TRAIN_PER  = 80
N_TEST_PER   = 25

# (name, model_type, aggregator, use_zk, num_utilities, n_train_per_override, lr_override)
METHODS = [
    ("Federated MLP",         "mlp_perbus", "fedavg",     False, 5, None, None),
    ("Federated MLP-flat",    "mlp_flat",   "fedavg",     False, 5, None, None),
    ("FedAvg-GATv2",          "gatv2",      "fedavg",     False, 5, None, None),
    ("Multi-Krum-GATv2",      "gatv2",      "multikrum",  False, 5, None, None),
    ("MK-GATv2 + ZKP (ours)", "gatv2",      "multikrum",  True,  5, None, None),
    # Centralized: 1 utility with all data, lower lr to prevent divergence
    ("Centralized GATv2",     "gatv2",      "fedavg",     False, 1, 200,  3e-3),
]

def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "results", "exp1_main.json")
    if os.path.exists(out_path):
        with open(out_path) as fh:
            out = json.load(fh)
        # Drop centralized runs since they're broken; rerun with new lr
        out["runs"] = [r for r in out["runs"] if r["method"] != "Centralized GATv2"]
        done = set((r["case"], r["seed"], r["method"]) for r in out["runs"])
        print(f"Resume: {len(out['runs'])} runs already saved (excluding broken Centralized).", flush=True)
    else:
        out = {"config": {"rounds": ROUNDS, "n_train_per": N_TRAIN_PER,
                           "n_test_per": N_TEST_PER, "seeds": SEEDS},
               "runs": []}
        done = set()
    t_total = time.perf_counter()
    for case in CASES:
        for seed in SEEDS:
            for (name, mtype, agg, usezk, nu, ntp_override, lr_override) in METHODS:
                key = (case, seed, name)
                if key in done:
                    continue
                ntp = ntp_override if ntp_override else N_TRAIN_PER
                nte = N_TEST_PER * 5 if nu == 1 else N_TEST_PER
                lr = lr_override if lr_override else 1e-2
                t0 = time.perf_counter()
                print(f"[{case} seed={seed}] {name} ...", flush=True)
                res = run_federated(
                    case=case, model_type=mtype,
                    num_utilities=nu,
                    n_train_per=ntp, n_test_per=nte,
                    rounds=ROUNDS, seed=seed,
                    aggregator=agg, f_byzantine=0,
                    byz_attack="honest",
                    use_zk_bound=usezk, tau=8.0,
                    hidden=8, heads=4, lr=lr, pos_weight=8.0,
                    eval_every=EVAL_EVERY, verbose=False,
                )
                wall = time.perf_counter() - t0
                m = res["final_metric"]
                print(f"  -> F1={m['f1']:.3f} prec={m['precision']:.3f} "
                      f"rec={m['recall']:.3f} auc={m['auc']:.3f}  ({wall:.1f}s)", flush=True)
                out["runs"].append({
                    "case": case, "seed": seed, "method": name,
                    "model_type": mtype, "aggregator": agg, "use_zk": usezk,
                    "num_utilities": nu,
                    "final_metric": m,
                    "history": res["history"],
                    "num_params": res["num_params"],
                    "proof_size_bytes": res["proof_size_bytes"],
                    "comm_bytes_per_client_per_round": res["comm_bytes_per_client_per_round"],
                    "wall_s": wall,
                })
                done.add(key)
                with open(out_path, "w") as fh:
                    json.dump(out, fh, indent=1, default=float)
    print(f"\nDone. Total: {time.perf_counter()-t_total:.1f}s")
    return out

if __name__ == "__main__":
    main()
