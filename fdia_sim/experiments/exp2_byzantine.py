"""
Experiment 2 — Byzantine robustness sweep (Sec. VIII-D of paper).

Three aggregators × five Byzantine fractions × two attack classes (sign-flip
and unbounded-norm scale-up) × one seed on case33bw.
The unbounded attack is the precise failure mode that Multi-Krum alone
cannot handle but MK + ZKP gradient bound does.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.runner import run_federated

CASE         = "case33bw"
SEEDS        = [42]
ROUNDS       = 15
N_TRAIN_PER  = 60
N_TEST_PER   = 25
NUM_UTILS    = 10
F_VALUES     = [0, 1, 2, 3, 4]
ATTACKS      = ["signflip", "unbounded"]
AGGREGATORS  = [
    ("FedAvg",        "fedavg",     False),
    ("Multi-Krum",    "multikrum",  False),
    ("MK + ZKP",      "multikrum",  True),
]

def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "results", "exp2_byzantine.json")
    if os.path.exists(out_path):
        with open(out_path) as fh:
            out = json.load(fh)
        done = set((r["seed"], r["attack"], r["f"], r["method"]) for r in out["runs"])
        print(f"Resume: {len(out['runs'])} runs saved.", flush=True)
    else:
        out = {"config": {"case": CASE, "seeds": SEEDS, "rounds": ROUNDS,
                           "num_utilities": NUM_UTILS, "f_values": F_VALUES,
                           "attacks": ATTACKS},
               "runs": []}
        done = set()
    t_total = time.perf_counter()
    for seed in SEEDS:
        for attack in ATTACKS:
            for f in F_VALUES:
                for (name, agg, usezk) in AGGREGATORS:
                    key = (seed, attack, f, name)
                    if key in done:
                        continue
                    t0 = time.perf_counter()
                    print(f"[seed={seed} {attack} f={f}] {name} ...", flush=True)
                    res = run_federated(
                        case=CASE, model_type="gatv2",
                        num_utilities=NUM_UTILS,
                        n_train_per=N_TRAIN_PER, n_test_per=N_TEST_PER,
                        rounds=ROUNDS, seed=seed,
                        aggregator=agg, f_byzantine=f,
                        byz_attack=attack if f > 0 else "honest",
                        use_zk_bound=usezk, tau=8.0,
                        hidden=8, heads=4, lr=1e-2, pos_weight=8.0,
                        eval_every=ROUNDS, verbose=False,
                    )
                    m = res["final_metric"]
                    wall = time.perf_counter() - t0
                    print(f"  -> F1={m['f1']:.3f} auc={m['auc']:.3f}  "
                          f"({wall:.1f}s)  rejected_total={sum(len(r) for r in res['history']['rejected_per_round'])}",
                          flush=True)
                    out["runs"].append({
                        "seed": seed, "attack": attack, "f": f,
                        "method": name, "aggregator": agg, "use_zk": usezk,
                        "final_metric": m,
                        "byz_ids": res["byz_ids"],
                        "wall_s": wall,
                        "rejected_per_round": res["history"]["rejected_per_round"],
                    })
                    done.add(key)
                    with open(out_path, "w") as fh:
                        json.dump(out, fh, indent=1, default=float)
    print(f"\nDone. Total: {time.perf_counter()-t_total:.1f}s")

if __name__ == "__main__":
    main()
