"""Decoding module."""
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, isspmatrix_csr
from numba import cuda
import math
import warnings
from utils import binaryproduct, gausselimination, check_random_state, incode, _bitsandnodes
from numba.cuda import current_context


def decode(H, y, snr, maxiter=100):
    """Decode a Gaussian noise corrupted n bits message using BP algorithm."""
    m, n = H.shape

    # Convert H to a sparse matrix
    H_sparse = validate_sparse_matrix(H)

    # Extract bits and nodes
    bits_hist, bits_values, nodes_hist, nodes_values = _bitsandnodes(H)

    # Convert to NumPy arrays
    bits_hist = np.array(bits_hist, dtype=np.int64)
    nodes_hist = np.array(nodes_hist, dtype=np.int64)

    # Flatten `bits_values` and `nodes_values`
    bits_values_flat = np.concatenate(bits_values).astype(np.int64)
    nodes_values_flat = np.concatenate(nodes_values).astype(np.int64)

    # Convert to CuPy arrays
    bits_hist_cp = cp.asarray(bits_hist, dtype=cp.int32)
    nodes_hist_cp = cp.asarray(nodes_hist, dtype=cp.int32)
    bits_values_flat_cp = cp.asarray(bits_values_flat, dtype=cp.int32)
    nodes_values_flat_cp = cp.asarray(nodes_values_flat, dtype=cp.int32)

    var = 10 ** (-snr / 10)

    if y.ndim == 1:
        y = y[:, None]

    # Initialization
    Lc = 2 * y / var
    _, n_messages = y.shape

    # Convert Lc to CuPy array
    Lc_cp = cp.asarray(Lc, dtype=cp.float64)

    # Run solver
    Lq, Lr, L_posteriori = run_cuda_solver(
        bits_hist_cp, bits_values_flat_cp, nodes_hist_cp, nodes_values_flat_cp, Lc_cp, maxiter
    )

    # Decode the message
    decoded_message = (L_posteriori <= 0).astype(
        np.int32)  # Convert to NumPy array for output

    if not incode(H, decoded_message):
        warnings.warn(
            "Decoding stopped before convergence. You may want to increase maxiter."
        )
    return decoded_message.squeeze()


def validate_array(array, expected_dim, expected_dtype, name):
    """
    Validates the input array for dimension and dtype compatibility.

    Parameters:
    - array: The array to validate.
    - expected_dim: Expected number of dimensions.
    - expected_dtype: Expected data type of the array.
    - name: Name of the array (for error messages).

    Raises:
    - ValueError: If the array does not meet the requirements.
    """
    if not isinstance(array, cp.ndarray):
        raise ValueError(f"{name} must be a CuPy array.")
    if array.ndim != expected_dim:
        raise ValueError(f"{name} must have {
                         expected_dim} dimensions, but got {array.ndim}.")
    if array.dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {
                         expected_dtype}, but got {array.dtype}.")


def validate_sparse_matrix(matrix):
    """
    Ensures the input is a CSR sparse matrix with dtype float64.

    Parameters:
    - matrix: Input sparse matrix (NumPy or CuPy array).

    Returns:
    - csr_matrix: A validated and possibly converted CSR sparse matrix.
    """
    # If it's not already a CSR matrix, convert it
    if not isspmatrix_csr(matrix):
        if isinstance(matrix, np.ndarray):  # Convert NumPy array to CuPy CSR
            matrix = cp.array(matrix, dtype=cp.float64)  # Convert to CuPy
            matrix = csr_matrix(matrix)  # Convert to CSR
        elif isinstance(matrix, cp.ndarray):  # Convert CuPy dense array to CSR
            matrix = csr_matrix(matrix.astype(cp.float64))
        else:
            raise ValueError(
                "Input must be a NumPy array, CuPy array, or a CSR sparse matrix.")
    elif matrix.dtype != cp.float64:  # Ensure the matrix is of type float64
        matrix = matrix.astype(cp.float64)
    return matrix


def run_cuda_solver(bits_hist, bits_values, nodes_hist, nodes_values, Lc, n_iter):
    """
    Runs the CUDA LogBP solver for sparse LDPC decoding.

    Parameters:
    - bits_hist: Histogram of bits per row.
    - bits_values: Values representing row connections.
    - nodes_hist: Histogram of nodes per column.
    - nodes_values: Values representing column connections.
    - Lc: Channel likelihoods.
    - n_iter: Number of iterations for the LogBP algorithm.

    Returns:
    - Lq: Likelihoods for variable nodes.
    - Lr: Likelihoods for check nodes.
    - L_posteriori: Posterior likelihoods.
    """
    # Validate inputs
    validate_array(bits_hist, 1, cp.int32, "bits_hist")
    validate_array(bits_values, 1, cp.int32, "bits_values")
    validate_array(nodes_hist, 1, cp.int32, "nodes_hist")
    validate_array(nodes_values, 1, cp.int32, "nodes_values")
    validate_array(Lc, 2, cp.float64, "Lc")

    # Dimensions
    m, n = bits_hist.size, nodes_hist.size
    n_messages = Lc.shape[1]

    # Allocate device memory
    d_bits_hist = cp.asarray(bits_hist)
    d_bits_values = cp.asarray(bits_values)
    d_nodes_hist = cp.asarray(nodes_hist)
    d_nodes_values = cp.asarray(nodes_values)
    d_Lc = cp.asarray(Lc)
    d_Lq = cp.zeros((m, n, n_messages), dtype=cp.float64)
    d_Lr = cp.zeros((m, n, n_messages), dtype=cp.float64)
    d_L_posteriori = cp.zeros((n, n_messages), dtype=cp.float64)

    # Configure CUDA blocks
    threads_per_block = (32, 32)
    blocks_per_grid = (
        (m + threads_per_block[0] - 1) // threads_per_block[0],
        (n + threads_per_block[1] - 1) // threads_per_block[1],
    )

    # Run CUDA kernel
    logbp_cuda[blocks_per_grid, threads_per_block](
        d_bits_hist, d_bits_values, d_nodes_hist, d_nodes_values,
        d_Lc, d_Lq, d_Lr, n_iter, d_L_posteriori
    )

    # Copy results back to host
    Lq = cp.asnumpy(d_Lq)
    Lr = cp.asnumpy(d_Lr)
    L_posteriori = cp.asnumpy(d_L_posteriori)
    return Lq, Lr, L_posteriori


@cuda.jit
def logbp_cuda(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr, n_iter, L_posteriori):
    """
    CUDA implementation of the LogBP solver.
    """
    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    ty = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if tx < bits_hist.shape[0]:
        start_idx = bits_values[tx]
        ff = bits_hist[tx]
        tanh_product = 1.0
        for i in range(ff):
            idx = start_idx + i
            scalar_val = 0.5 * \
                Lc[idx, 0] if n_iter == 0 else 0.5 * Lq[tx, idx, 0]
            tanh_val = math.tanh(scalar_val)
            tanh_product *= tanh_val

        for i in range(ff):
            idx = start_idx + i
            scalar_val = 0.5 * \
                Lc[idx, 0] if n_iter == 0 else 0.5 * Lq[tx, idx, 0]
            tanh_val = math.tanh(scalar_val)
            result = (1 + tanh_product / tanh_val) / \
                (1 - tanh_product / tanh_val)
            Lr[tx, idx, 0] = math.log(max(result, 1e-10))

    if ty < nodes_hist.shape[0]:
        start_idx = nodes_values[ty]
        ff = nodes_hist[ty]
        for i in range(ff):
            idx = start_idx + i
            Lq[idx, ty, 0] = Lc[ty, 0]
            for j in range(ff):
                if j != i:
                    Lq[idx, ty, 0] += Lr[start_idx + j, ty, 0]

        posterior = Lc[ty, 0]
        for i in range(ff):
            idx = start_idx + i
            posterior += Lr[idx, ty, 0]
        L_posteriori[ty, 0] = posterior


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
