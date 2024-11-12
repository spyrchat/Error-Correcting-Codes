def is_erasure(received_signal, threshold):
    """
    Function to check if the received signal falls within the erasure region.

    Args:
    - received_signal (float): The value of the received signal.
    - threshold (float): The threshold for defining the erasure region.

    Returns:
    - bool: Returns True if the signal is an erasure, otherwise False.
    """
    if abs(received_signal) < threshold:
        return True  # Erasure
    return False  # Not an erasure


import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix


def generate_ldpc_matrix(n, k, row_weight=3, col_weight=6):
    """
    Generates a regular LDPC parity-check matrix with given row and column weights.

    Args:
    - n (int): Length of codeword.
    - k (int): Number of information bits.
    - row_weight (int): Number of 1s in each row (check node degree).
    - col_weight (int): Number of 1s in each column (variable node degree).

    Returns:
    - H (csr_matrix): Parity-check matrix of size (n-k) x n.
    """
    m = n - k
    H = sparse_random(m, n, density=row_weight / n, format='csr')
    H.data[:] = 1  # Set all non-zero entries to 1 to form a binary matrix
    return H


def encode_ldpc(H, message):
    """
    Encodes a message using an LDPC code defined by the parity-check matrix H.

    Args:
    - H (csr_matrix): Parity-check matrix.
    - message (np.array): Message bits.

    Returns:
    - codeword (np.array): LDPC encoded codeword.
    """
    # Simplified encoding for demonstration; in practice, you would solve Hx = 0 for x.
    k = message.size
    codeword = np.concatenate((message, np.zeros(H.shape[0])))
    return codeword  # Placeholder, actual encoding is more complex.


def simulate_ldpc_bpsk_with_erasures(H, num_bits, erasure_threshold, num_iterations=100):
    """
    Simulates an LDPC code with BPSK and erasures, calculating symbol error rate and bit rate.

    Args:
    - H (csr_matrix): LDPC parity-check matrix.
    - num_bits (int): Number of information bits per iteration.
    - erasure_threshold (float): Threshold for defining erasure regions.
    - num_iterations (int): Number of iterations in the simulation.

    Returns:
    - (float, float): Symbol error rate and bit rate.
    """
    k = num_bits  # Number of information bits
    n = H.shape[1]  # Length of the codeword
    symbol_errors = 0
    transmitted_bits = 0

    for _ in range(num_iterations):
        # Step 1: Generate a random message and encode with LDPC
        message = np.random.randint(0, 2, k)
        codeword = encode_ldpc(H, message)

        # Step 2: Modulate with BPSK (0 -> +1, 1 -> -1)
        transmitted_signal = 1 - 2 * codeword  # BPSK: 0 -> +1, 1 -> -1

        # Step 3: Add noise and create received signal
        noise = np.random.normal(0, 0.5, n)  # Assume standard deviation of noise is 0.5
        received_signal = transmitted_signal + noise

        # Step 4: Apply erasure regions
        erasures = np.abs(received_signal) < erasure_threshold
        decoded_signal = np.where(received_signal > 0, 0, 1)  # Hard-decision BPSK demodulation

        # Step 5: Calculate errors and update counters
        errors = (decoded_signal != codeword) & ~erasures
        symbol_errors += np.sum(errors)
        transmitted_bits += np.sum(~erasures)

    # Calculate symbol error rate and bit rate
    symbol_error_rate = symbol_errors / (n * num_iterations)
    bit_rate = transmitted_bits / (n * num_iterations)

    return symbol_error_rate, bit_rate


# Simulation parameters
n = 100  # Length of codeword
k = 50  # Number of information bits
H = generate_ldpc_matrix(n, k)  # Generate a regular LDPC parity-check matrix
erasure_threshold = 0.5  # Threshold for defining erasure regions

# Run simulation
symbol_error_rate, bit_rate = simulate_ldpc_bpsk_with_erasures(H, k, erasure_threshold)
print(f"Symbol Error Rate: {symbol_error_rate}")
print(f"Bit Rate: {bit_rate}")


