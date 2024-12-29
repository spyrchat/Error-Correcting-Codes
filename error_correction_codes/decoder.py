"""Decoding module."""
import numpy as np
import warnings

from numba import njit, int64, types, float64
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
        solver = _logbp_numba_regular  # Regular LDPC
    else:
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




output_type_log2 = types.Tuple((float64[:, :, :], float64[:, :, :],
                               float64[:, :]))


# Define the output type for Numba-compiled functions
output_type_log2 = types.Tuple((float64[:, :, :], float64[:, :, :], float64[:, :]))

@njit(output_type_log2(int64[:], int64[:], int64[:], int64[:], float64[:, :],
                       float64[:, :, :], float64[:, :, :], int64), cache=True)
def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter):
    """Perform inner LogBP solver for irregular LDPC matrices."""
    m, n, n_messages = Lr.shape

    bits_counter = 0  # Keeps track of bit connections in flattened arrays
    for i in range(m):
        ff = bits_hist[i]  # Degree of check node i
        ni = bits_values[bits_counter: bits_counter + ff]  # Connected variable nodes
        bits_counter += ff
        
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

    nodes_counter = 0  # Keeps track of node connections in flattened arrays
    for j in range(n):
        ff = nodes_hist[j]  # Degree of variable node j
        mj = nodes_values[nodes_counter: nodes_counter + ff]  # Connected check nodes
        nodes_counter += ff

        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]
            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # Compute posterior log-likelihood ratios
    L_posteriori = np.zeros((n, n_messages))
    nodes_counter = 0
    for j in range(n):
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

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
    """Compute the original `n_bits` message from a `n_code` codeword `x`.

    Parameters
    ----------
    tG: array (n_code, n_bits) coding matrix tG.
    x: array (n_code,) decoded codeword of length `n_code`.

    Returns
    -------
    message: array (n_bits,). Original binary message.

    """
    n, k = tG.shape

    if len(x) != n:
        raise ValueError(f"Inconsistent dimensions: x has {len(x)} elements, but tG has {n} columns.")

    if k != n - tG.shape[0]:
        raise ValueError(f"Inconsistent dimensions: tG has {tG.shape[0]} rows but k={k}")


    if len(tG) != k:
        raise ValueError(f"Inconsistent dimensions: tG has {len(tG)} rows but k={k}")

    rtG, rx = gausselimination(tG, x)

    message = np.zeros(k).astype(int)

    # Extract message bits
    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= binaryproduct(rtG[i, list(range(i + 1, k))],
                                    message[list(range(i + 1, k))])

    return abs(message)

