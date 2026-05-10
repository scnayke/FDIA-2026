"""
Haber-Stornetta hash-chain audit ledger.

Each appended record commits to: (round id, global model hash, accepted
client gradient hashes, rejected client identifiers, prior chain head).
Anchoring to a permissioned blockchain is out of scope for the
small-scale validation; here we measure only the local append cost.
"""
import hashlib
import json
import time


class HashChain:
    def __init__(self):
        self.head = b"\x00" * 32
        self.records = []  # (record_bytes, head_after)

    def append(self, round_id: int, model_hash: bytes,
               accepted_hashes: list[bytes], rejected_ids: list[int]):
        record = json.dumps({
            "r":  int(round_id),
            "m":  model_hash.hex(),
            "a":  [h.hex() for h in accepted_hashes],
            "x":  list(rejected_ids),
        }, sort_keys=True).encode()
        h = hashlib.sha256(self.head + record).digest()
        self.head = h
        self.records.append((record, h))
        return h

    def __len__(self):
        return len(self.records)


def hash_vec(vec_bytes: bytes) -> bytes:
    return hashlib.sha256(vec_bytes).digest()


def time_anchor(round_id, model_hash, accepted_hashes, rejected_ids):
    """Return wall-clock seconds to append one record (for latency table)."""
    chain = HashChain()
    t0 = time.perf_counter()
    chain.append(round_id, model_hash, accepted_hashes, rejected_ids)
    return time.perf_counter() - t0
