import numpy as np
import warnings
from utils import incode, _bitsandnodes, gausselimination, binaryproduct

from numba import njit, int64, types, float64


import numpy as np


def decode(H, y, snr, maxiter=1000):
    """Decode a Gaussian noise corrupted n bits message using BP algorithm."""

    m, n = H.shape

    # Ensure bits_hist, bits_values, nodes_hist, nodes_values are NumPy arrays
    bits_hist, bits_values, nodes_hist, nodes_values = _bitsandnodes(H)

    # Convert lists to NumPy arrays
    bits_hist = np.array(bits_hist, dtype=np.int64)
    nodes_hist = np.array(nodes_hist, dtype=np.int64)

    # Flatten nested lists into 1D NumPy arrays
    bits_values_flat = np.concatenate(
        [np.array(b, dtype=np.int64) for b in bits_values])
    nodes_values_flat = np.concatenate(
        [np.array(n, dtype=np.int64) for n in nodes_values])

    # Pick correct solver (Regular or Irregular)
    _n_bits = np.unique(H.sum(0))
    _n_nodes = np.unique(H.sum(1))

    if len(_n_bits) == 1 and len(_n_nodes) == 1:
        solver = _logbp_numba_regular
    else:
        solver = _logbp_numba

    # Ensure y is a NumPy array
    y = np.array(y, dtype=np.float64)

    var = 10 ** (-snr / 10)

    if y.ndim == 1:
        y = y[:, None]

    Lc = 2 * y / var
    _, n_messages = y.shape

    Lq = np.zeros((m, n, n_messages), dtype=np.float64)
    Lr = np.zeros((m, n, n_messages), dtype=np.float64)

    for n_iter in range(maxiter):
        Lq, Lr, L_posteriori = solver(
            bits_hist, bits_values_flat, nodes_hist, nodes_values_flat,
            Lc, Lq, Lr, n_iter
        )
        x = np.array(L_posteriori <= 0, dtype=int)
        if incode(H, x):
            break

    return x.squeeze()


output_type_log2 = types.Tuple((float64[:, :, :], float64[:, :, :],
                               float64[:, :]))


@njit(output_type_log2(int64[:], int64[:], int64[:], int64[:], float64[:, :],
                       float64[:, :, :],  float64[:, :, :], int64), cache=True)
def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr,
                 n_iter):
    """Perform inner ext LogBP solver."""
    m, n, n_messages = Lr.shape
    # step 1 : Horizontal

    bits_counter = 0
    nodes_counter = 0
    for i in range(m):
        # ni = bits[i]
        ff = bits_hist[i]
        ni = bits_values[bits_counter: bits_counter + ff]
        bits_counter += ff
        for j in ni:
            nij = ni[:]

            X = np.ones(n_messages)
            if n_iter == 0:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lc[nij[kk]])
            else:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lq[i, nij[kk]])
            num = 1 + X
            denom = 1 - X
            for ll in range(n_messages):
                if num[ll] == 0:
                    Lr[i, j, ll] = -1
                elif denom[ll] == 0:
                    Lr[i, j, ll] = 1
                else:
                    Lr[i, j, ll] = np.log(num[ll] / denom[ll])

    # step 2 : Vertical
    for j in range(n):
        # mj = nodes[j]
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]

            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # LLR a posteriori:
    L_posteriori = np.zeros((n, n_messages))
    nodes_counter = 0
    for j in range(n):
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

    return Lq, Lr, L_posteriori


@njit(output_type_log2(int64[:], int64[:, :], int64[:], int64[:, :],
                       float64[:, :], float64[:, :, :],  float64[:, :, :],
                       int64), cache=True)
def _logbp_numba_regular(bits_hist, bits_values, nodes_hist, nodes_values, Lc,
                         Lq, Lr, n_iter):
    """Perform inner ext LogBP solver."""
    m, n, n_messages = Lr.shape
    # step 1 : Horizontal
    for i in range(m):
        ni = bits_values[i]
        for j in ni:
            nij = ni[:]

            X = np.ones(n_messages)
            if n_iter == 0:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lc[nij[kk]])
            else:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lq[i, nij[kk]])
            num = 1 + X
            denom = 1 - X
            for ll in range(n_messages):
                if num[ll] == 0:
                    Lr[i, j, ll] = -1
                elif denom[ll] == 0:
                    Lr[i, j, ll] = 1
                else:
                    Lr[i, j, ll] = np.log(num[ll] / denom[ll])

    # step 2 : Vertical
    for j in range(n):
        mj = nodes_values[j]
        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]

            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # LLR a posteriori:
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

    rtG, rx = gausselimination(tG, x)

    message = np.zeros(k).astype(int)

    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= binaryproduct(rtG[i, list(range(i+1, k))],
                                    message[list(range(i+1, k))])

    return abs(message)
