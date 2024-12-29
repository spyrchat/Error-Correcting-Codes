"""Decoding module."""
import numpy as np
import warnings

from numba import njit, int64, types, float64, prange
from utils import binaryproduct, gausselimination, check_random_state, incode, _bitsandnodes


def decode(H, y, snr, maxiter=1000):
    """Decode a Gaussian noise corrupted n bits message using BP algorithm."""
    m, n = H.shape

    # Extract bits and nodes
    bits_hist, bits_values, nodes_hist, nodes_values = _bitsandnodes(H)

    # Convert to NumPy arrays for numba compatibility
    bits_hist = np.array(bits_hist, dtype=np.int64)
    nodes_hist = np.array(nodes_hist, dtype=np.int64)

    # Flatten `bits_values` and `nodes_values` for compatibility
    bits_values_flat = np.concatenate(bits_values).astype(np.int64)
    nodes_values_flat = np.concatenate(nodes_values).astype(np.int64)

    # Determine solver based on LDPC matrix properties
    _n_bits = np.unique(H.sum(axis=0))
    _n_nodes = np.unique(H.sum(axis=1))

    if len(_n_bits) == 1 and len(_n_nodes) == 1:
        print("Regular LDPC matrix detected.")
        solver = _logbp_numba_regular  # Regular LDPC
    else:
        print("Irregular LDPC matrix detected.")
        solver = _logbp_numba  # Irregular LDPC

    var = 10 ** (-snr / 10)

    if y.ndim == 1:
        y = y[:, None]

    # Initialization
    Lc = 2 * y / var
    _, n_messages = y.shape

    Lq = np.zeros(shape=(m, n, n_messages), dtype=np.float64)
    Lr = np.zeros(shape=(m, n, n_messages), dtype=np.float64)

    for n_iter in range(maxiter):
        # Updated solver call with only 8 arguments
        Lq, Lr, L_posteriori = solver(
            bits_hist, bits_values_flat,
            nodes_hist, nodes_values_flat,
            Lc, Lq, Lr, n_iter
        )
        x = np.array(L_posteriori <= 0).astype(int)
        product = incode(H, x)
        if product:
            break

    if n_iter == maxiter - 1:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x.squeeze()


# Define the output type for the _logbp_numba function
output_type_log2 = types.Tuple((float64[:, :, :], float64[:, :, :], float64[:, :]))

@njit(output_type_log2(int64[:], int64[:], int64[:], int64[:], float64[:, :],
                       float64[:, :, :], float64[:, :, :], int64), cache=True, parallel=True)
def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter):
    """Perform inner LogBP solver for irregular LDPC matrices."""
    m, n, n_messages = Lr.shape

    bits_counter = 0
    print(f"Starting LogBP for {m} check nodes and {n} variable nodes")
    for i in prange(m):  # Parallelize over check nodes
        if i % 1000 == 0:
            print(f"Processing check node {i}/{m}...")

        ff = bits_hist[i]
        ni = bits_values[bits_counter: bits_counter + ff]
        bits_counter += ff

        for j in ni:
            nij = ni[:]
            tanh_buffer = np.tanh(0.5 * (Lc[nij] if n_iter == 0 else Lq[i, nij]))
            tanh_product = np.ones(n_messages)
            for kk in range(ff):
                if nij[kk] != j:
                    tanh_product *= tanh_buffer[kk]

            num = 1 + tanh_product
            denom = 1 - tanh_product
            Lr[i, j] = np.log(np.clip(num / denom, 1e-10, 1e10))

    nodes_counter = 0
    for j in prange(n):  # Parallelize over variable nodes
        if j % 1000 == 0:
            print(f"Processing variable node {j}/{n}...")

        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff

        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]
            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    L_posteriori = np.zeros((n, n_messages))
    nodes_counter = 0
    for j in prange(n):
        if j % 1000 == 0:
            print(f"Calculating posterior for node {j}/{n}...")

        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

    print("LogBP computation completed.")
    return Lq, Lr, L_posteriori


@njit(output_type_log2(int64[:], int64[:, :], int64[:], int64[:, :],
                       float64[:, :], float64[:, :, :], float64[:, :, :], int64), cache=True)
def _logbp_numba_regular(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter):
    """Perform inner LogBP solver for regular LDPC matrices."""
    m, n, n_messages = Lr.shape

    for i in range(m):
        ni = bits_values[i]  # Connected variable nodes

        for j in ni:
            nij = ni[:]

            # Compute the product of tanh messages
            X = np.ones(n_messages)
            for kk in range(len(nij)):
                if nij[kk] != j:
                    X *= np.tanh(0.5 * (Lc[nij[kk]] if n_iter == 0 else Lq[i, nij[kk]]))

            num = 1 + X
            denom = 1 - X
            Lr[i, j] = np.log(np.clip(num / denom, 1e-10, 1e10))

    for j in range(n):
        mj = nodes_values[j]  # Connected check nodes

        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]
            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # Compute posterior log-likelihood ratios
    L_posteriori = np.zeros((n, n_messages))
    for j in range(n):
        mj = nodes_values[j]
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

    return Lq, Lr, L_posteriori


def get_message(tG, x):
    """Compute the original `n_bits` message from a `n_code` codeword `x`."""
    n, k = tG.shape
    if len(x) != n:
        raise ValueError(f"Inconsistent dimensions: x has {len(x)} elements, but tG has {n} rows.")

    if k > len(x):
        raise ValueError(f"Inconsistent dimensions: tG requires {k} columns, but x has {len(x)} elements.")

    print(f"Input x dimensions: {len(x)}, tG dimensions: {tG.shape}")  # Debugging

    # Gaussian elimination to reduce the system
    rtG, rx = gausselimination(tG, x)

    # Ensure rx has at least `k` elements before processing
    if len(rx) < k:
        raise ValueError(f"rx has {len(rx)} elements, expected at least {k}.")

    # Extract message bits
    message = np.zeros(k).astype(int)
    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= binaryproduct(rtG[i, list(range(i + 1, k))],
                                    message[list(range(i + 1, k))])

    return abs(message)
