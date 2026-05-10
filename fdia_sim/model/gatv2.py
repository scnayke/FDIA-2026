"""
GATv2 layer and model implemented in autograd.numpy.

Implements the fixed-attention variant of Brody, Alon, Yahav (ICLR 2022).
For each layer:

    score(i, j) = a^T LeakyReLU(W_src x_i + W_dst x_j)
    alpha(i, j) = softmax_{j in N(i)} score(i, j)
    h_i'        = ELU( sum_{j in N(i)} alpha(i, j) * (W_dst x_j) )

Multi-head: H heads in parallel; intermediate layers concatenate, output
layer averages. Self-loops are added to edge_index so each node attends
to itself.

The whole forward function is plain numpy (via autograd's wrapper) so
backward is obtained via autograd.grad without manual derivation.

We keep the model deliberately small (2 GATv2 layers, 4 heads of width
8, then a 2-layer MLP head) so it stays in the ~2K-3K parameter range
and trains in reasonable wall-clock time on a CPU.
"""
import autograd.numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp
import numpy as np


# ---------- helpers ----------------------------------------------------

def add_self_loops(edge_index, num_nodes):
    """edge_index: (2, E) numpy. Adds (i, i) for each i. Returns a new (2, E') array."""
    self_e = np.arange(num_nodes, dtype=edge_index.dtype)
    sl = np.stack([self_e, self_e], axis=0)
    return np.concatenate([edge_index, sl], axis=1)


def init_glorot(shape, rng):
    fan_in, fan_out = shape[0], shape[-1]
    s = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-s, s, size=shape).astype(np.float32)


# ---------- model parameter init --------------------------------------

def init_params(F_in, hidden, heads, F_out_class=1, rng=None):
    """Two GATv2 layers + 2-layer MLP head producing per-node logit.

    Shapes:
      W_src1, W_dst1: (heads, F_in, hidden)
      a1:             (heads, hidden)
      W_src2, W_dst2: (heads, heads*hidden, hidden)   # in dim = heads*hidden after concat
      a2:             (heads, hidden)
      W_mlp1:         (hidden, hidden)
      b_mlp1:         (hidden,)
      W_mlp2:         (hidden, F_out_class)
      b_mlp2:         (F_out_class,)
    Layer 1 concatenates heads (output dim = heads*hidden).
    Layer 2 averages heads (output dim = hidden).
    """
    rng = rng or np.random.default_rng(0)
    p = {}
    p["W_src1"] = init_glorot((heads, F_in,            hidden), rng)
    p["W_dst1"] = init_glorot((heads, F_in,            hidden), rng)
    p["a1"]     = init_glorot((heads, hidden),                  rng)

    p["W_src2"] = init_glorot((heads, heads * hidden, hidden), rng)
    p["W_dst2"] = init_glorot((heads, heads * hidden, hidden), rng)
    p["a2"]     = init_glorot((heads, hidden),                  rng)

    p["W_mlp1"] = init_glorot((hidden, hidden),                 rng)
    p["b_mlp1"] = np.zeros((hidden,), dtype=np.float32)
    p["W_mlp2"] = init_glorot((hidden, F_out_class),            rng)
    p["b_mlp2"] = np.zeros((F_out_class,), dtype=np.float32)
    return p


def num_params(p):
    return sum(int(np.asarray(v).size) for v in p.values())


# ---------- forward ----------------------------------------------------

def _segment_softmax(scores_eh, dst_e, num_nodes):
    """Per-destination softmax over edges, separately for each head.
    scores_eh: (E, H), dst_e: (E,) numpy ints. Returns alpha (E, H)."""
    H = scores_eh.shape[1]
    # numerically stable: per-destination max, then exp / per-destination sum
    maxd = anp.full((num_nodes, H), -1e9)
    for h in range(H):
        s_h = scores_eh[:, h]
        # maximum scatter
        # we use anp.maximum.at via numpy fallback because autograd doesn't
        # ship .at; instead build it from segment-sum using a one-hot
        pass
    # Simpler implementation: subtract per-destination max computed in plain numpy
    # (the max op is a constant w.r.t. autograd anyway since it's used only for stability).
    # Then exp/sum is fully differentiable.
    sc_np = np.asarray(scores_eh._value if hasattr(scores_eh, "_value") else scores_eh)
    # If we don't want to rely on private, compute max in numpy from a detached copy:
    # autograd treats numpy arrays as constants.
    if hasattr(scores_eh, "_value"):
        sc_np = np.asarray(scores_eh._value)
    else:
        sc_np = np.asarray(scores_eh)
    # per-destination max
    maxd_np = np.full((num_nodes, H), -1e9, dtype=np.float32)
    for h in range(H):
        for e in range(sc_np.shape[0]):
            d = dst_e[e]
            if sc_np[e, h] > maxd_np[d, h]:
                maxd_np[d, h] = sc_np[e, h]
    # broadcast subtract (autograd-safe)
    s_centered = scores_eh - maxd_np[dst_e]
    e_s = anp.exp(s_centered)            # (E, H)
    # per-destination sum via segment_sum implemented as boolean dot
    # sum_d[d, h] = sum_{e: dst_e[e]=d} e_s[e, h]
    # build one-hot index matrix M (num_nodes, E) then sum_d = M @ e_s
    # for small graphs this is fine.
    M = np.zeros((num_nodes, sc_np.shape[0]), dtype=np.float32)
    for e in range(sc_np.shape[0]):
        M[dst_e[e], e] = 1.0
    sum_d = anp.dot(M, e_s)              # (num_nodes, H)
    alpha = e_s / (sum_d[dst_e] + 1e-12)
    return alpha


def _segment_sum(values_eh, dst_e, num_nodes):
    """Sum over edges grouped by destination. values_eh: (E, H, F)."""
    E, H, F = values_eh.shape
    # build (num_nodes, E) selector then einsum
    M = np.zeros((num_nodes, E), dtype=np.float32)
    for e in range(E):
        M[dst_e[e], e] = 1.0
    # M @ values reshape: (num_nodes, H, F)
    flat = values_eh.reshape(E, H * F)
    out = anp.dot(M, flat)
    return out.reshape(num_nodes, H, F)


def gatv2_layer(x, edge_index, W_src, W_dst, a, leaky=0.2):
    """One GATv2 layer.
    x: (N, F_in)
    edge_index: (2, E) numpy int (with self-loops added beforehand)
    W_src, W_dst: (H, F_in, F_h)
    a: (H, F_h)
    Returns h: (N, H, F_h)  — caller will concat or average heads.
    """
    src = edge_index[0]; dst = edge_index[1]
    N = x.shape[0]
    H_ = W_src.shape[0]
    F_h = W_src.shape[2]
    # message values per head: m_{e,h,:} = (W_dst[h]) @ x[src[e]]   (NB: GATv2 paper uses W applied to dst neighbour x_j for the *value*; we follow the standard PyG convention where the per-head transformation is shared.)
    # We apply the SAME W_dst as the value matrix.
    # Compute per-head linear projections of x (N, H, F_h)
    h_src = anp.einsum("nf,hfd->nhd", x, W_src)   # contributions from x_i (target side, per-head)
    h_dst = anp.einsum("nf,hfd->nhd", x, W_dst)   # contributions from x_j (source-neighbour side, per-head)
    # Edge-wise pre-activation: e_{e,h,:} = h_src[dst_e, h, :] + h_dst[src_e, h, :]
    pre = h_src[dst] + h_dst[src]                # (E, H, F_h)
    # LeakyReLU
    pre = anp.where(pre > 0, pre, leaky * pre)
    # Score: contract last dim with a
    scores = anp.einsum("ehd,hd->eh", pre, a)    # (E, H)
    # Softmax per destination
    alpha = _segment_softmax(scores, dst, N)     # (E, H)
    # Aggregate values (use h_dst[src] as the value)
    val = h_dst[src]                              # (E, H, F_h)
    weighted = val * alpha[:, :, None]           # (E, H, F_h)
    out = _segment_sum(weighted, dst, N)          # (N, H, F_h)
    return out


def forward(p, x, edge_index_self):
    """End-to-end forward, returns per-node logits (N, 1)."""
    h1 = gatv2_layer(x, edge_index_self, p["W_src1"], p["W_dst1"], p["a1"])  # (N, H, F_h)
    # ELU and concat heads
    h1 = anp.where(h1 > 0, h1, anp.exp(h1) - 1.0)
    N, H_, F_h = h1.shape
    h1c = h1.reshape(N, H_ * F_h)                        # (N, H*F_h)
    h2 = gatv2_layer(h1c, edge_index_self, p["W_src2"], p["W_dst2"], p["a2"])  # (N, H, F_h)
    h2 = anp.where(h2 > 0, h2, anp.exp(h2) - 1.0)
    h2m = anp.mean(h2, axis=1)                            # average heads → (N, F_h)
    # MLP head
    z  = anp.dot(h2m, p["W_mlp1"]) + p["b_mlp1"]
    z  = anp.where(z > 0, z, anp.exp(z) - 1.0)            # ELU
    out = anp.dot(z, p["W_mlp2"]) + p["b_mlp2"]            # (N, 1)
    return out


# ---------- loss + optim ----------------------------------------------

def bce_with_logits(logits, y, pos_weight=1.0):
    """Binary cross-entropy with logits, y in {0,1}.
    logits: (N, 1), y: (N,). Logits are clamped to [-30, 30] to keep
    autograd's exp() from overflowing on outlier batches; this clamp
    has zero effect on a well-conditioned model since BCE saturates
    long before |z| reaches 30."""
    z = anp.clip(logits[:, 0], -30.0, 30.0)
    pos = y * pos_weight
    loss_pos = pos * anp.maximum(z, 0) - pos * z + pos * anp.log1p(anp.exp(-anp.abs(z)))
    loss_neg = (1 - y) * (anp.maximum(z, 0) + anp.log1p(anp.exp(-anp.abs(z))))
    return anp.mean(loss_pos + loss_neg)


def predict_proba(p, x, edge_index_self):
    z = forward(p, x, edge_index_self)[:, 0]
    return 1.0 / (1.0 + np.exp(-z))


# ---------- adam optimizer in numpy -----------------------------------

class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
    def step(self, params, grads):
        self.t += 1
        out = {}
        for k in params:
            g = np.asarray(grads[k], dtype=np.float32)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            out[k] = (params[k] - self.lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)
        return out


def params_to_vec(p):
    """Flatten parameter dict into a single 1-D float32 vector (deterministic key order)."""
    keys = sorted(p.keys())
    parts = [np.asarray(p[k], dtype=np.float32).ravel() for k in keys]
    return np.concatenate(parts)


def params_diff(p_new, p_old):
    """Element-wise difference dict (gradient update)."""
    return {k: (p_new[k] - p_old[k]).astype(np.float32) for k in p_old}


def apply_diff(p, delta):
    """Return p + delta."""
    return {k: (p[k] + delta[k]).astype(np.float32) for k in p}


def vec_to_params(vec, like):
    out = {}
    cursor = 0
    for k in sorted(like.keys()):
        n = int(np.asarray(like[k]).size)
        out[k] = vec[cursor:cursor + n].reshape(like[k].shape).astype(np.float32)
        cursor += n
    return out


def clip_diff(delta_vec, tau):
    """If ||delta_vec||_2 > tau, scale to tau."""
    nrm = float(np.linalg.norm(delta_vec))
    if nrm <= tau or nrm == 0:
        return delta_vec, nrm
    return (delta_vec * (tau / nrm)).astype(np.float32), nrm
