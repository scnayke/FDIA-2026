"""
Grid simulation utilities.

Loads IEEE distribution / benchmark cases via pandapower, runs AC power flow
under perturbed load profiles to generate realistic measurement vectors,
and constructs the measurement Jacobian H so that unobservable FDIA attack
vectors a = H c can be crafted in the column space of H per Liu, Ning,
Reiter (ACM TISSEC 2011).

We use pandapower's case33bw (Baran-Wu 33-bus radial distribution feeder)
as a small distribution feeder, and case118 (IEEE 118-bus) as a larger
benchmark — both are standard, peer-reviewed test cases shipped with
pandapower 3.4.
"""
import numpy as np
import pandapower as pp
import pandapower.networks as pn


def load_net(name: str):
    """Return a fresh pandapower net by short name."""
    if name == "case33bw":
        return pn.case33bw()
    elif name == "case118":
        return pn.case118()
    else:
        raise ValueError(f"unknown case {name}")


def edge_index(net):
    """Return (2, num_edges) numpy array of bus index pairs from net.line and net.trafo.
    Buses are reindexed to a contiguous 0..N-1 range using net.bus.index."""
    bus_id_to_idx = {b: i for i, b in enumerate(net.bus.index)}
    edges = []
    for _, row in net.line.iterrows():
        u = bus_id_to_idx[row.from_bus]
        v = bus_id_to_idx[row.to_bus]
        edges.append((u, v))
        edges.append((v, u))   # treat as undirected for GNN message passing
    if hasattr(net, "trafo") and len(net.trafo) > 0:
        for _, row in net.trafo.iterrows():
            u = bus_id_to_idx[row.hv_bus]
            v = bus_id_to_idx[row.lv_bus]
            edges.append((u, v))
            edges.append((v, u))
    return np.array(edges, dtype=np.int64).T  # (2, E)


def edge_features(net):
    """Per-edge features: r, x, length-or-1, type-flag (0=line,1=trafo)."""
    feats = []
    bus_id_to_idx = {b: i for i, b in enumerate(net.bus.index)}
    # lines
    for _, row in net.line.iterrows():
        r = float(row.get("r_ohm_per_km", 0.0)) * float(row.get("length_km", 1.0))
        x = float(row.get("x_ohm_per_km", 0.0)) * float(row.get("length_km", 1.0))
        ln = float(row.get("length_km", 1.0))
        feats.append([r, x, ln, 0.0])
        feats.append([r, x, ln, 0.0])
    if hasattr(net, "trafo") and len(net.trafo) > 0:
        for _, row in net.trafo.iterrows():
            feats.append([0.0, 0.0, 1.0, 1.0])
            feats.append([0.0, 0.0, 1.0, 1.0])
    arr = np.asarray(feats, dtype=np.float32)
    # standardize columns to zero-mean unit-var
    mu = arr.mean(0, keepdims=True)
    sd = arr.std(0, keepdims=True) + 1e-6
    return (arr - mu) / sd


def run_one(net, scale: float = 1.0):
    """Scale all loads uniformly by `scale`, run AC PF, return per-bus state.
    Returns a dict with arrays length N (number of buses):
      vm   : voltage magnitude (p.u.)
      va   : voltage angle (radians)
      p_mw : net real injection (gen - load) in MW
      q_mvar : net reactive injection in MVAr
    Raises pandapower.LoadflowNotConverged on failure."""
    saved = net.load["p_mw"].copy(), net.load["q_mvar"].copy()
    net.load["p_mw"]  = saved[0] * scale
    net.load["q_mvar"] = saved[1] * scale
    try:
        pp.runpp(net, numba=False, init="flat")
    finally:
        net.load["p_mw"]  = saved[0]
        net.load["q_mvar"] = saved[1]

    bus_id_to_idx = {b: i for i, b in enumerate(net.bus.index)}
    N = len(net.bus)
    vm = np.zeros(N); va = np.zeros(N); p = np.zeros(N); q = np.zeros(N)
    for bid, row in net.res_bus.iterrows():
        i = bus_id_to_idx[bid]
        vm[i] = row.vm_pu
        va[i] = row.va_degree * np.pi / 180.0
        # res_bus.p_mw is the net injection convention used by pandapower
        p[i]  = row.p_mw
        q[i]  = row.q_mvar
    return {"vm": vm, "va": va, "p_mw": p, "q_mvar": q}


def per_bus_features(state, net):
    """Pack per-bus 6-dim feature vector used by the GNN.
    Columns: |V|, angle(V), P, Q, type_flag, load_flag.
    type_flag = 1 for slack/ext_grid bus, 0 otherwise.
    load_flag = 1 if there is a load attached, 0 otherwise."""
    bus_id_to_idx = {b: i for i, b in enumerate(net.bus.index)}
    N = len(net.bus)
    type_flag = np.zeros(N, dtype=np.float32)
    if len(net.ext_grid) > 0:
        for _, row in net.ext_grid.iterrows():
            type_flag[bus_id_to_idx[row.bus]] = 1.0
    load_flag = np.zeros(N, dtype=np.float32)
    for _, row in net.load.iterrows():
        load_flag[bus_id_to_idx[row.bus]] = 1.0
    feats = np.stack([state["vm"], state["va"], state["p_mw"], state["q_mvar"],
                       type_flag, load_flag], axis=1).astype(np.float32)
    return feats


def standardize(X, stats=None):
    """Column-wise zero-mean/unit-var. Returns (Xz, stats). If stats given, applies it."""
    if stats is None:
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True) + 1e-6
        stats = (mu, sd)
    return (X - stats[0]) / stats[1], stats


# ---------------- Measurement Jacobian and FDIA -----------------------

def measurement_jacobian(net, state):
    """Construct a *DC-approximation* measurement Jacobian H mapping bus
    angle perturbations c (length N) to changes in power-flow measurements
    P_inj at every bus (length N). H = B', the susceptance matrix used
    in DC power flow. This is a textbook simplification (see Wood &
    Wollenberg, *Power Generation, Operation, and Control*) and is the
    standard model used by Liu-Ning-Reiter to characterise unobservable
    FDIA. Diagonal H_{ii} = sum_j 1/x_{ij}; off-diagonal H_{ij} = -1/x_{ij}.

    Returns dense numpy array of shape (N, N)."""
    bus_id_to_idx = {b: i for i, b in enumerate(net.bus.index)}
    N = len(net.bus)
    H = np.zeros((N, N), dtype=np.float64)
    # only line susceptances (skip trafos with x=0 in standard cases)
    for _, row in net.line.iterrows():
        u = bus_id_to_idx[row.from_bus]
        v = bus_id_to_idx[row.to_bus]
        x = float(row.get("x_ohm_per_km", 0.0)) * float(row.get("length_km", 1.0))
        if x <= 0: continue
        b_uv = 1.0 / x
        H[u, u] += b_uv
        H[v, v] += b_uv
        H[u, v] -= b_uv
        H[v, u] -= b_uv
    # transformers contribute too
    if hasattr(net, "trafo") and len(net.trafo) > 0:
        for _, row in net.trafo.iterrows():
            u = bus_id_to_idx[row.hv_bus]
            v = bus_id_to_idx[row.lv_bus]
            xk = float(row.get("vk_percent", 5.0)) / 100.0  # fallback
            sn = float(row.get("sn_mva", 100.0))
            if xk <= 0 or sn <= 0: continue
            b_uv = 1.0 / xk
            H[u, u] += b_uv
            H[v, v] += b_uv
            H[u, v] -= b_uv
            H[v, u] -= b_uv
    return H


def craft_fdia(H, sparsity=5, magnitude=0.05, rng=None):
    """Return (a, c) such that a = H @ c with ||c||_0 <= sparsity. The vector
    `a` lies in the column space of H, so adding it to power-injection
    measurements preserves the residual. Magnitudes are in p.u. radians
    for c (state perturbation)."""
    rng = rng or np.random.default_rng()
    N = H.shape[0]
    c = np.zeros(N)
    # exclude bus 0 (slack reference) from attack
    targets = rng.choice(np.arange(1, N), size=min(sparsity, N - 1), replace=False)
    c[targets] = rng.uniform(-magnitude, magnitude, size=len(targets))
    a = H @ c
    return a, c


def label_compromised_buses(c, threshold=1e-9):
    """Return a per-bus binary label vector: 1 if |c_i| > threshold else 0."""
    return (np.abs(c) > threshold).astype(np.int64)
