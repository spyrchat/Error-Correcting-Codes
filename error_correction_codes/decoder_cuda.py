"""Decoding module."""
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, isspmatrix_csr
from numba import cuda
import math
import warnings
from utils import binaryproduct, gausselimination, check_random_state, incode, _bitsandnodes


def decode(H, y, snr, maxiter=1000):
    """Decode a Gaussian noise corrupted n bits message using BP algorithm."""
    m, n = H.shape

    # Convert H to a sparse matrix
    H_sparse = ensure_compatible_sparse(H)

    # Extract bits and nodes
    bits_hist, bits_values, nodes_hist, nodes_values = _bitsandnodes(H)

    # Convert to NumPy arrays for CUDA compatibility
    bits_hist = np.array(bits_hist, dtype=np.int64)
    nodes_hist = np.array(nodes_hist, dtype=np.int64)

    # Flatten `bits_values` and `nodes_values` for compatibility
    bits_values_flat = np.concatenate(bits_values).astype(np.int64)
    nodes_values_flat = np.concatenate(nodes_values).astype(np.int64)

    var = 10 ** (-snr / 10)

    if y.ndim == 1:
        y = y[:, None]

    # Initialization
    Lc = 2 * y / var
    _, n_messages = y.shape

    # Run solver
    Lq, Lr, L_posteriori = logbp_cuda(H_sparse, Lc, maxiter)

    x = np.array(L_posteriori <= 0).astype(int)
    if not incode(H, x):
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x.squeeze()


@cuda.jit
def logbp_cuda_sparse(data, indices, indptr, Lc, Lq, Lr, n_iter, L_posteriori):
    """
    CUDA implementation of the LogBP solver for sparse matrices.
    """
    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # Thread index for check nodes
    ty = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y  # Thread index for variable nodes

    # Horizontal Processing: Check Nodes -> Variable Nodes
    if tx < indptr.shape[0] - 1:  # Ensure within bounds
        start_idx = indptr[tx]
        end_idx = indptr[tx + 1]
        ff = end_idx - start_idx

        tanh_product = 1.0
        for i in range(start_idx, end_idx):  # Calculate tanh product
            idx = indices[i]
            scalar_val = 0.5 * Lc[idx, 0] if n_iter == 0 else 0.5 * Lq[tx, idx, 0]
            tanh_val = math.tanh(scalar_val)
            tanh_product *= tanh_val

        for i in range(start_idx, end_idx):  # Update Lr
            idx = indices[i]
            scalar_val = 0.5 * Lc[idx, 0] if n_iter == 0 else 0.5 * Lq[tx, idx, 0]
            tanh_val = math.tanh(scalar_val)
            result = (1 + tanh_product / tanh_val) / (1 - tanh_product / tanh_val)
            Lr[tx, idx, 0] = math.log(max(result, 1e-10))

    # Posterior Log-Likelihood Ratios
    if ty < Lc.shape[0]:  # Ensure within bounds
        start_idx = indptr[ty]
        end_idx = indptr[ty + 1]
        posterior = Lc[ty, 0]
        for i in range(start_idx, end_idx):
            idx = indices[i]
            posterior += Lr[idx, ty, 0]
        L_posteriori[ty, 0] = posterior


def ensure_compatible_sparse(matrix):
    """
    Ensures the input is a CSR sparse matrix with a supported type (float64).
    """
    if not isspmatrix_csr(matrix):
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Input must be a NumPy array or a CSR matrix.")
        # Convert to float64 and then to CSR
        matrix = csr_matrix(matrix.astype(np.float64))
    elif matrix.dtype != np.float64:
        # Convert data type to float64 if necessary
        matrix = matrix.astype(np.float64)
    return matrix


def ensure_compatible_dense(array, dtype=np.float64):
    """
    Ensures the input is a dense array with the correct dtype.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if array.dtype != dtype:
        array = array.astype(dtype)
    return array


def logbp_cuda(sparse_matrix, Lc, n_iter):
    """
    Runs the CUDA LogBP solver for sparse matrices.
    """
    m, n = sparse_matrix.shape
    n_messages = Lc.shape[1]

    # Convert sparse matrix to CSR format with compatible type
    sparse_matrix = ensure_compatible_sparse(sparse_matrix)

    # Extract CSR components
    data = sparse_matrix.data
    indices = sparse_matrix.indices
    indptr = sparse_matrix.indptr

    # Allocate device memory
    d_data = cuda.to_device(data)
    d_indices = cuda.to_device(indices)
    d_indptr = cuda.to_device(indptr)
    d_Lc = cuda.to_device(Lc)
    d_Lq = cuda.to_device(cp.zeros((m, n, n_messages), dtype=cp.float64))
    d_Lr = cuda.to_device(cp.zeros((m, n, n_messages), dtype=cp.float64))
    d_L_posteriori = cuda.to_device(cp.zeros((n, n_messages), dtype=cp.float64))

    # Configure CUDA blocks
    threads_per_block = (32, 32)
    blocks_per_grid = ((m + threads_per_block[0] - 1) // threads_per_block[0],
                       (n + threads_per_block[1] - 1) // threads_per_block[1])

    # Run CUDA kernel
    logbp_cuda_sparse[blocks_per_grid, threads_per_block](
        d_data, d_indices, d_indptr, d_Lc, d_Lq, d_Lr, n_iter, d_L_posteriori
    )

    # Copy results back to host
    Lq = d_Lq.copy_to_host()
    Lr = d_Lr.copy_to_host()
    L_posteriori = d_L_posteriori.copy_to_host()
    return Lq, Lr, L_posteriori


def get_message(tG, x):
    """Compute the original `n_bits` message from a `n_code` codeword `x`."""
    n, k = tG.shape
    if len(x) != n:
        raise ValueError(f"Inconsistent dimensions: x has {len(x)} elements, but tG has {n} rows.")

    if k > len(x):
        raise ValueError(f"Inconsistent dimensions: tG requires {k} columns, but x has {len(x)} elements.")

    # Gaussian elimination to reduce the system
    rtG, rx = gausselimination(tG, x)

    # Extract message bits
    message = np.zeros(k).astype(int)
    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= binaryproduct(rtG[i, list(range(i + 1, k))],
                                    message[list(range(i + 1, k))])

    return abs(message)
