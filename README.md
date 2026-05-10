# Byzantine-Robust ZK-Verifiable Federated GNN for FDIA — Deliverables

This bundle contains the publication-ready paper, full LaTeX source, simulation
code, and raw experimental results for "A Byzantine-Robust, Zero-Knowledge-
Verifiable Federated Graph Neural Network for False Data Injection Attack
Detection in Smart Grid Distribution Systems".

## Contents

```
deliverables/
├── byzantine_fdia_paper.pdf      The compiled paper (9 pages, IEEEtran)
├── paper/
│   ├── main.tex                  LaTeX source (default class: IEEEtran conf)
│   ├── references.bib            Bibliography with 70+ verified citations
│   ├── fig_convergence.pdf       Fig. 2 — F1 vs round
│   ├── fig_byzantine.pdf         Fig. 3 — Byzantine robustness sweep
│   └── fig_latency.pdf           Fig. 4 — Per-round latency breakdown
├── fdia_sim/                     Full simulation code (CPU-only, pure NumPy)
│   ├── grid/                     Grid simulation (pandapower) + FDIA crafting
│   ├── model/                    GATv2 (autograd-NumPy) and MLP baselines
│   ├── fl/                       Federated learning core + aggregators + eval
│   ├── attacks/                  Byzantine attack simulators
│   ├── zk/                       Bulletproofs timing model
│   ├── audit/                    Hash-chain audit ledger
│   └── experiments/              5 experiment scripts + figure generator
└── results/                      Raw measurement JSONs
    ├── exp1_main.json            Detection performance (12 runs)
    ├── exp2_byzantine.json       Byzantine sweep (30 runs)
    ├── exp3_latency.json         Per-round latency
    ├── exp4_tau.json             Norm-bound sensitivity
    └── exp5_n.json               Number-of-clients sensitivity
```

## Compiling the paper (IEEEtran, default)

```
cd paper/
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Requires TeX Live 2022+ with `IEEEtran`, `algorithm`, `algpseudocode`,
`booktabs`, `tikz`, `pgfplots` (>= 1.18), `hyperref`. On Debian/Ubuntu:
`sudo apt install texlive-publishers texlive-pictures texlive-latex-extra`.

## Switching to Springer LNCS

1. Replace `\documentclass[conference]{IEEEtran}` with `\documentclass{llncs}`.
2. Remove `\IEEEoverridecommandlockouts` and the `IEEEkeywords` environment;
   replace with `\keywords{FDIA \and Federated Learning \and GNN \and ZKP \and Smart Grid}`.
3. Replace the IEEEtran author block with:
   ```
   \author{Saurabh B. Koravi \and Reshma Sonar}
   \institute{MIT World Peace University, Pune, India \\
     \email{saurabh.koravi@mitwpu.edu.in,reshma.sonar@mitwpu.edu.in}}
   ```
4. Replace `\bibliographystyle{IEEEtran}` with `\bibliographystyle{splncs04}`.
5. Compile as above.

## Reproducing the simulation results

The simulation runs on a CPU-only Ubuntu 22.04 VM with no GPU dependency.

### Dependencies

```
pip install pandapower==3.4 autograd numpy scipy scikit-learn matplotlib
```

(No PyTorch required — the GATv2 forward pass and backward gradient are
implemented in pure NumPy via `autograd`, which keeps the entire system
runnable in 3.9 GB of disk.)

### Running

```
cd fdia_sim/

# Detection performance (≈ 30 min)
python3 experiments/exp1_main.py

# Byzantine robustness sweep (≈ 45 min)
python3 experiments/exp2_byzantine.py

# Per-round latency (≈ 4 min)
python3 experiments/exp3_latency.py

# Tau sensitivity (≈ 6 min)
python3 experiments/exp4_tau.py

# n sensitivity (≈ 10 min)
python3 experiments/exp5_n.py

# Generate figures
python3 experiments/make_figures.py
```

All scripts are resumable: rerunning skips runs already saved in the
corresponding JSON. Total cost end-to-end: ≈ 1.6 hours on the spec'd VM.

## Where each contribution lives in the paper

- **C1 Decoupled architecture** — Sec. IV (System Design), Fig. 1, Theorem 2.
- **C2 Compact ZK gradient-bound** — Sec. VI-B, Lemma 1.
- **C3 Theoretical analysis** — Sec. VII (Theorems 1–2, Lemma 1, Proposition 1).
- **C4 Empirical evaluation** — Sec. VIII (Tables II–V, Figs. 2–4).

## Honest scope notes

- **Test system:** The paper uses the IEEE 33-bus radial distribution feeder
  (Baran-Wu, shipped as `case33bw` in pandapower) as the canonical small
  distribution test case. The IEEE 37/123 distribution feeders are not
  bundled in pandapower 3.4; the IEEE 118-bus benchmark was attempted as
  a substitute but did not converge under the small-parameter GATv2 within
  the rounds budget. This is documented openly in Sec. VIII-A and Sec. IX.

- **ZKP primitive:** The Bulletproofs prove and verify times are from a
  calibrated timing model fitted to published `dalek-bulletproofs` 4.0
  benchmarks; the paper does not implement the cryptographic primitive
  end-to-end. A production deployment would link `dalek-bulletproofs` via
  FFI; the rest of the system (Multi-Krum, audit ledger, model training)
  is implemented in full and is unchanged by that swap. Disclosed in
  Sec. IX.

- **Reproducibility:** All seeds (42, 1234), hyperparameters, and code paths
  are committed in the simulation source. The results JSONs in `results/`
  were generated from the included scripts.

## Acknowledgement of AI-assistance

Code, paper text, and figures in this bundle were produced in collaboration
with Claude (Anthropic) under the supervision of the corresponding authors.
All numerical results are from real simulation runs on the spec'd CPU VM
and can be regenerated by following the reproduction instructions above.
