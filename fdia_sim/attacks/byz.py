"""
Byzantine client behaviours used in the Sec. VIII robustness sweep.

Each attack is a function of (honest_delta_dict, rng) -> malicious_delta_dict.
We deliberately keep them simple and well-characterised in the literature
so that reviewers can map our results onto Fang et al. (USENIX 2020)
and Bagdasaryan et al. (AISTATS 2020).
"""
import numpy as np
from model.gatv2 import params_to_vec, vec_to_params


def attack_signflip(delta, rng):
    return {k: -v.copy() for k, v in delta.items()}


def attack_gauss(delta, rng, sigma=1.0):
    out = {}
    for k, v in delta.items():
        out[k] = (v + rng.normal(0, sigma, size=v.shape)).astype(np.float32)
    return out


def attack_unbounded_scale(delta, rng, scale=20.0):
    """Lift gradient norm by a large multiplicative factor — the precise
    failure-mode of Multi-Krum that the ZK norm-bound proof closes."""
    return {k: (scale * v).astype(np.float32) for k, v in delta.items()}


def attack_label_flip_apply(y):
    """Used at the client level: flip the labels before training. Returns
    flipped y_list. Should be called inside the malicious client's
    local_train, not at aggregation time."""
    return [1 - yy for yy in y]


# Higher-level wrapper used by the experiment runner -------------------

ATTACK_FNS = {
    "signflip":   lambda d, rng: attack_signflip(d, rng),
    "gauss":      lambda d, rng: attack_gauss(d, rng, sigma=1.0),
    "unbounded":  lambda d, rng: attack_unbounded_scale(d, rng, scale=20.0),
    "labelflip":  None,  # handled at client level (training-time)
    "honest":     lambda d, rng: d,
}
