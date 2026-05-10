"""
Non-graph federated baselines.

* MLP: a 2-hidden-layer MLP that classifies each bus from its own 6-dim
  feature vector independently (i.e. ignores topology). This is the
  apples-to-apples non-graph baseline.

* LSTM-flat: We do not have a temporal sequence in this static setup,
  so "LSTM" here is meaningless; instead we provide a slightly-deeper
  MLP on the flattened (N * F) snapshot vector and project back to N
  per-bus logits. This emulates what a "flatten the grid into a 1-D
  vector and apply a sequential model" baseline would do, and is the
  standard non-graph reference in the FDIA-detection literature
  (cf. He, Mendis, Wei IEEE TSG 2017).
"""
import autograd.numpy as anp
import numpy as np


def init_mlp_per_bus(F_in, hidden, F_out=1, rng=None):
    rng = rng or np.random.default_rng(0)
    s = lambda fi, fo: np.sqrt(6.0 / (fi + fo))
    return {
        "W1": rng.uniform(-s(F_in, hidden), s(F_in, hidden), (F_in, hidden)).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "W2": rng.uniform(-s(hidden, hidden), s(hidden, hidden), (hidden, hidden)).astype(np.float32),
        "b2": np.zeros(hidden, dtype=np.float32),
        "W3": rng.uniform(-s(hidden, F_out), s(hidden, F_out), (hidden, F_out)).astype(np.float32),
        "b3": np.zeros(F_out, dtype=np.float32),
    }


def forward_mlp_per_bus(p, x, edge_index_self):
    """Treats each bus independently: x: (N, F_in) -> logits (N, 1)."""
    h = anp.dot(x, p["W1"]) + p["b1"]
    h = anp.where(h > 0, h, anp.exp(h) - 1)
    h = anp.dot(h, p["W2"]) + p["b2"]
    h = anp.where(h > 0, h, anp.exp(h) - 1)
    return anp.dot(h, p["W3"]) + p["b3"]


def init_mlp_flat(F_in, N, hidden, rng=None):
    rng = rng or np.random.default_rng(0)
    D = F_in * N
    s = lambda fi, fo: np.sqrt(6.0 / (fi + fo))
    return {
        "W1": rng.uniform(-s(D, hidden), s(D, hidden), (D, hidden)).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "W2": rng.uniform(-s(hidden, hidden), s(hidden, hidden), (hidden, hidden)).astype(np.float32),
        "b2": np.zeros(hidden, dtype=np.float32),
        "W3": rng.uniform(-s(hidden, N), s(hidden, N), (hidden, N)).astype(np.float32),
        "b3": np.zeros(N, dtype=np.float32),
    }


def forward_mlp_flat(p, x, edge_index_self):
    """Flatten the (N, F_in) snapshot, run a deep MLP, return per-bus logits."""
    flat = x.reshape(1, -1)                # (1, N*F_in)
    h = anp.dot(flat, p["W1"]) + p["b1"]
    h = anp.where(h > 0, h, anp.exp(h) - 1)
    h = anp.dot(h, p["W2"]) + p["b2"]
    h = anp.where(h > 0, h, anp.exp(h) - 1)
    out = anp.dot(h, p["W3"]) + p["b3"]    # (1, N)
    return out.reshape(-1, 1)               # (N, 1)
