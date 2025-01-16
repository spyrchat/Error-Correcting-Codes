import math
from numba import cuda
from cupyx.scipy.sparse import csr_matrix, isspmatrix_csr
import cupy as cp
from erasure_channel_encoding_irregular import simulate_irregular_ldpc_erasure_correction
from decoder import get_message
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


def test_get_message():
    # Example parameters
    tG = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])  # Corrected: k=3
    # Decoded codeword with length matching tG's columns
    x = np.array([1, 0, 1])

    # Get original message
    message = get_message(tG, x)

    # Assertions
    assert message.shape == (3,), f"Message shape mismatch: {message.shape}"
    print(f"get_message passed. message={message}")


test_get_message()


@cuda.jit
def logbp_cuda_sparse(data, indices, indptr, Lc, Lq, Lr, n_iter, L_posteriori):
    """
    CUDA implementation of the LogBP solver for sparse matrices.
    """
    tx = cuda.threadIdx.x + cuda.blockIdx.x * \
        cuda.blockDim.x  # Thread index for check nodes
    ty = cuda.threadIdx.y + cuda.blockIdx.y * \
        cuda.blockDim.y  # Thread index for variable nodes

    # Horizontal Processing: Check Nodes -> Variable Nodes
    if tx < indptr.shape[0] - 1:  # Ensure within bounds
        start_idx = indptr[tx]
        end_idx = indptr[tx + 1]
        ff = end_idx - start_idx

        tanh_product = 1.0
        for i in range(start_idx, end_idx):  # Calculate tanh product
            idx = indices[i]
            scalar_val = 0.5 * \
                Lc[idx, 0] if n_iter == 0 else 0.5 * Lq[tx, idx, 0]
            tanh_val = math.tanh(scalar_val)
            tanh_product *= tanh_val

        for i in range(start_idx, end_idx):  # Update Lr
            idx = indices[i]
            scalar_val = 0.5 * \
                Lc[idx, 0] if n_iter == 0 else 0.5 * Lq[tx, idx, 0]
            tanh_val = math.tanh(scalar_val)
            result = (1 + tanh_product / tanh_val) / \
                (1 - tanh_product / tanh_val)
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
        if not isinstance(matrix, cp.ndarray):
            raise ValueError("Input must be a CuPy array or a CSR matrix.")
        # Convert to float64 and then to CSR
        matrix = csr_matrix(matrix.astype(cp.float64))
    elif matrix.dtype != cp.float64:
        # Convert data type to float64 if necessary
        matrix = matrix.astype(cp.float64)
    return matrix


def ensure_compatible_dense(array, dtype=cp.float64):
    """
    Ensures the input is a dense array with the correct dtype.
    """
    if not isinstance(array, cp.ndarray):
        raise ValueError("Input must be a CuPy array.")
    if array.dtype != dtype:
        array = array.astype(dtype)
    return array


def validate_inputs(sparse_matrix, Lc):
    """
    Validates and prepares inputs for the solver.
    """
    sparse_matrix = ensure_compatible_sparse(sparse_matrix)
    Lc = ensure_compatible_dense(Lc, dtype=cp.float64)
    return sparse_matrix, Lc


def run_cuda_solver_sparse(sparse_matrix, Lc, n_iter):
    """
    Runs the CUDA LogBP solver for sparse matrices.
    """
    m, n = sparse_matrix.shape
    n_messages = Lc.shape[1]

    # Convert sparse matrix to CSR format with compatible type
    if not isspmatrix_csr(sparse_matrix):
        sparse_matrix = csr_matrix(sparse_matrix.astype(cp.float64))

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
    d_L_posteriori = cuda.to_device(
        cp.zeros((n, n_messages), dtype=cp.float64))

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


def test_solver_correctness():
    """
    Validates the correctness of the CUDA LogBP solver with sparse matrices.
    """
    print("Testing correctness of the CUDA-accelerated LogBP solver with sparse matrices...")

    # Example sparse matrix creation
    dense_matrix = cp.random.randint(0, 2, size=(1000, 1000), dtype=cp.int32)
    sparse_matrix = csr_matrix(dense_matrix.astype(
        cp.float64))  # Convert to float64

    # Generate Lc as a dense matrix
    Lc = cp.random.randn(sparse_matrix.shape[1], 10).astype(cp.float64)

    # Run the solver
    try:
        Lq, Lr, L_posteriori = run_cuda_solver_sparse(
            sparse_matrix, Lc, n_iter=10)

        # Validate output shapes
        assert Lq.shape == (sparse_matrix.shape[0], sparse_matrix.shape[1], Lc.shape[1]), \
            f"Unexpected shape for Lq: {Lq.shape}"
        assert Lr.shape == (sparse_matrix.shape[0], sparse_matrix.shape[1], Lc.shape[1]), \
            f"Unexpected shape for Lr: {Lr.shape}"
        assert L_posteriori.shape == (sparse_matrix.shape[1], Lc.shape[1]), \
            f"Unexpected shape for L_posteriori: {L_posteriori.shape}"

        print("Correctness test passed successfully!")
    except Exception as e:
        print("Error during correctness testing:", str(e))
        raise


def test_solver_performance():
    """
    Tests the performance of the CUDA LogBP solver with a large sparse matrix.
    """
    print("Testing performance of the CUDA-accelerated LogBP solver with sparse matrices...")

    # Large sparse matrix creation
    dense_matrix = cp.random.randint(0, 2, size=(10000, 10000), dtype=cp.int32)
    sparse_matrix = csr_matrix(dense_matrix.astype(
        cp.float64))  # Convert to float64

    # Generate Lc as a dense matrix
    Lc = cp.random.randn(sparse_matrix.shape[1], 10).astype(cp.float64)

    # Measure execution time
    import time
    start_time = time.time()
    try:
        Lq, Lr, L_posteriori = run_cuda_solver_sparse(
            sparse_matrix, Lc, n_iter=10)
        end_time = time.time()

        print(f"Performance test completed in {
              end_time - start_time:.2f} seconds.")
    except Exception as e:
        print("Error during performance testing:", str(e))
        raise


# Run the tests
test_solver_correctness()
test_solver_performance()
