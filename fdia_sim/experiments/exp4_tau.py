"""
Experiment 4 — Sensitivity to norm bound tau (Sec. VIII.G of paper).

Sweep tau over {1.0, 2.0, 4.0, 8.0, 16.0, +inf} at f=0 on case33bw.
Records F1 and the fraction of HONEST gradients clipped to tau (the
"honest reject" rate when their natural norm > tau).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fl.runner import run_federated

TAU_VALUES = [1.0, 2.0, 4.0, 8.0, 16.0]
ROUNDS = 20
NU = 5
N_TR = 80
N_TE = 25
SEED = 42

def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "results", "exp4_tau.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        done = set(r["tau"] for r in out["runs"])
    else:
        out = {"config": {"rounds": ROUNDS, "n_train_per": N_TR}, "runs": []}
        done = set()
    t0 = time.perf_counter()
    for tau in TAU_VALUES:
        if tau in done:
            continue
        ts = time.perf_counter()
        print(f"[tau={tau}] running...", flush=True)
        res = run_federated(
            case="case33bw", model_type="gatv2",
            num_utilities=NU,
            n_train_per=N_TR, n_test_per=N_TE,
            rounds=ROUNDS, seed=SEED,
            aggregator="multikrum", f_byzantine=0, byz_attack="honest",
            use_zk_bound=True, tau=tau,
            hidden=8, heads=4, lr=1e-2, pos_weight=8.0,
            eval_every=ROUNDS, verbose=False,
        )
        m = res["final_metric"]
        # honest-reject rate: how many gradients across rounds had pre-clip
        # norm > tau (i.e. the bound was binding)
        pre = res["history"]["delta_norms_pre"]
        post = res["history"]["delta_norms_post"]
        bound_binding = 0; total = 0
        for round_pre, round_post in zip(pre, post):
            for p, q in zip(round_pre, round_post):
                total += 1
                if abs(p - q) > 1e-6:   # clipping kicked in
                    bound_binding += 1
        binding_pct = 100.0 * bound_binding / max(1, total)
        print(f"  -> F1={m['f1']:.3f} auc={m['auc']:.3f}  binding={binding_pct:.1f}% "
              f"({time.perf_counter()-ts:.1f}s)", flush=True)
        out["runs"].append({"tau": tau, "final_metric": m,
                            "honest_clip_pct": binding_pct,
                            "wall_s": time.perf_counter() - ts})
        done.add(tau)
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    print(f"\nDone. Total: {time.perf_counter()-t0:.1f}s")

if __name__ == "__main__":
    main()
