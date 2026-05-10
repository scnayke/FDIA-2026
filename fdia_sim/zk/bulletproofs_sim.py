"""
Bulletproofs zero-knowledge gradient-bound proof — TIMING MODEL only.

We do not reproduce a full Bulletproofs implementation in pure Python (a
real deployment would use the dalek-bulletproofs Rust crate via a Python
binding); for the small-scale latency study in Sec. VIII.E we use a
deterministic timing model calibrated to *published* benchmarks for
``dalek-bulletproofs`` 4.0 on commodity x86 CPUs:

  - Aggregate prover time:   ~ 1.4 + 1.05e-4 * d   seconds   for d coords
  - Verifier time (batched): ~ 0.011 + 4.5e-7 * d  seconds
  - Proof size:              32 * (2 * ceil(log2 d) + 9) bytes
                             ≈ 1.4 KB for d in 30k–60k

These coefficients are fit to the numbers reported in Bünz et al.
(IEEE S&P 2018, Table 3) and the open-source benchmarks in the dalek
repository (https://github.com/zkcrypto/bulletproofs).

A real implementation would replace this module with FFI calls; the
remainder of the system (Multi-Krum, audit ledger, etc.) is unchanged.

We also expose a deterministic ``simulate_proof_bytes`` that returns a
random-but-stable byte string of the right length so the audit ledger
test exercises the same byte-budget that production would.
"""
import math
import hashlib


# Fit to published dalek-bulletproofs / Bünz et al. 2018 numbers.
PROVE_BASE_S    = 1.40
PROVE_PER_COORD = 1.05e-4
VERIFY_BASE_S   = 0.011
VERIFY_PER_COORD = 4.5e-7


def proof_size_bytes(d: int) -> int:
    """Aggregate Bulletproofs range-proof size: 32 * (2 ceil(log2 d) + 9) bytes."""
    if d <= 1:
        return 32 * 9
    return 32 * (2 * int(math.ceil(math.log2(d))) + 9)


def estimated_prove_time_s(d: int) -> float:
    return PROVE_BASE_S + PROVE_PER_COORD * d


def estimated_verify_time_s(d: int) -> float:
    return VERIFY_BASE_S + VERIFY_PER_COORD * d


def simulate_proof_bytes(delta_vec_bytes: bytes, d: int) -> bytes:
    """Return a deterministic placeholder proof of the right length, derived
    from the delta vector hash for binding. Used only for byte-accounting."""
    L = proof_size_bytes(d)
    seed = hashlib.sha256(delta_vec_bytes).digest()
    out = b""
    counter = 0
    while len(out) < L:
        out += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    return out[:L]


def commitment_check(delta_vec, tau: float) -> bool:
    """The predicate the ZKP attests: ||delta||_2 <= tau."""
    import numpy as np
    return float(np.linalg.norm(delta_vec)) <= tau + 1e-6
