from utils import _bitsandnodes
import numpy as np  
def test_bitsandnodes():
    """Test the bitsandnodes function with a sample H matrix."""
    # Example parity-check matrix
    H = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0]
    ])

    bits_hist, bits_values, nodes_hist, nodes_values = _bitsandnodes(H)

    print("Bits histogram (row sums):", bits_hist)
    print("Bits values (row connections):", bits_values)
    print("Nodes histogram (column sums):", nodes_hist)
    print("Nodes values (column connections):", nodes_values)

# Run the test
test_bitsandnodes()

from decoder import get_message

def test_get_message():
    # Example parameters
    tG = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])  # Corrected: k=3
    x = np.array([1, 0, 1])  # Decoded codeword with length matching tG's columns

    # Get original message
    message = get_message(tG, x)

    # Assertions
    assert message.shape == (3,), f"Message shape mismatch: {message.shape}"
    print(f"get_message passed. message={message}")

test_get_message()

from erasure_channel_encoding_irregular import simulate_irregular_ldpc_erasure_correction

from numba import njit, int64, float64, prange
import numpy as np
from numba import types

# Define the output type for the _logbp_numba function
output_type_log2 = types.Tuple((float64[:, :, :], float64[:, :, :], float64[:, :]))

@njit(output_type_log2(int64[:], int64[:], int64[:], int64[:], float64[:, :],
                       float64[:, :, :], float64[:, :, :], int64), cache=True, parallel=True)
def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter):
    """Perform inner LogBP solver for irregular LDPC matrices."""
    m, n, n_messages = Lr.shape

    bits_counter = 0  # Keeps track of bit connections in flattened arrays
    for i in prange(m):  # Parallelize over check nodes
        ff = bits_hist[i]  # Degree of check node i
        ni = bits_values[bits_counter: bits_counter + ff]  # Connected variable nodes
        bits_counter += ff

        for j in ni:
            nij = ni[:]

            # Compute the product of tanh messages manually
            tanh_buffer = np.tanh(0.5 * (Lc[nij] if n_iter == 0 else Lq[i, nij]))
            tanh_product = np.ones(n_messages)
            for kk in range(ff):
                if nij[kk] != j:
                    tanh_product *= tanh_buffer[kk]  # Manual product without using axis

            num = 1 + tanh_product
            denom = 1 - tanh_product
            Lr[i, j] = np.log(np.clip(num / denom, 1e-10, 1e10))

    nodes_counter = 0  # Keeps track of node connections in flattened arrays
    for j in prange(n):  # Parallelize over variable nodes
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
    for j in prange(n):
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

    return Lq, Lr, L_posteriori

# Test function
def stress_test_logbp():
    print("Starting LogBP stress test...")

    # Generate large sparse LDPC matrices
    m, n = 10000, 20000  # 10,000 check nodes, 20,000 variable nodes
    n_messages = 10  # Simulate 10 messages

    bits_hist = np.random.randint(1, 10, size=m)  # Degrees for check nodes
    nodes_hist = np.random.randint(1, 10, size=n)  # Degrees for variable nodes

    bits_values = np.concatenate([np.random.choice(np.arange(n), h, replace=False) for h in bits_hist])
    nodes_values = np.concatenate([np.random.choice(np.arange(m), h, replace=False) for h in nodes_hist])

    Lc = np.random.randn(n, n_messages)  # Channel log-likelihoods
    Lq = np.zeros((m, n, n_messages))
    Lr = np.zeros((m, n, n_messages))

    # Convert integer arrays to int64 for compatibility
    bits_hist = bits_hist.astype(np.int64)
    nodes_hist = nodes_hist.astype(np.int64)
    bits_values = bits_values.astype(np.int64)
    nodes_values = nodes_values.astype(np.int64)

    # Perform one iteration of LogBP
    Lq, Lr, L_posteriori = _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter=0)


    # Validate output shapes
    assert Lq.shape == (m, n, n_messages), "Lq has incorrect shape!"
    assert Lr.shape == (m, n, n_messages), "Lr has incorrect shape!"
    assert L_posteriori.shape == (n, n_messages), "L_posteriori has incorrect shape!"

    print("Stress test completed successfully!")

# Run the stress test
stress_test_logbp()