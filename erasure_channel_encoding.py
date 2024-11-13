import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode, get_message

def simulate_ldpc_with_pyldpc(erasure_thresholds, n, d_v, d_c, noise_std=0.2, num_iterations=100):
    """
    Simulate LDPC encoding, transmission with noise and erasures, and decoding
    using pyldpc with Belief Propagation decoding.

    Args:
    - erasure_thresholds (list): List of erasure thresholds to test.
    - n (int): Length of the codeword.
    - d_v (int): Degree of variable nodes.
    - d_c (int): Degree of check nodes.
    - noise_std (float): Standard deviation of the Gaussian noise.
    - num_iterations (int): Number of iterations in the simulation.

    Returns:
    - ser_results (list): Symbol Error Rates (SER) for each erasure threshold.
    - bit_rate_results (list): Bit Rates for each erasure threshold.
    """
    # Generate LDPC code
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]  # Number of information bits based on generator matrix

    ser_results = []
    bit_rate_results = []

    for threshold in erasure_thresholds:
        symbol_errors = 0
        total_transmitted_bits = 0

        for _ in range(num_iterations):
            # Step 1: Generate a random message and encode it
            message = np.random.randint(0, 2, k)
            codeword = encode(G, message, snr=10)  # Encode with SNR for internal pyldpc purposes

            # Step 2: BPSK modulation
            transmitted_signal = 1 - 2 * codeword  # BPSK: 0 -> +1, 1 -> -1

            # Step 3: Add Gaussian noise
            noise = np.random.normal(0, noise_std, transmitted_signal.shape)
            received_signal = transmitted_signal + noise

            # Step 4: Apply erasure threshold
            erasures = np.abs(received_signal) < threshold
            received_signal[erasures] = np.nan  # Set erased values to NaN for pyldpc's decode function

            # Step 5: Decode with pyldpc's Belief Propagation
            decoded_codeword = decode(H, received_signal, snr=10, maxiter=50)
            decoded_message = get_message(G, decoded_codeword)

            # Step 6: Calculate errors
            errors = np.sum(decoded_message != message)
            symbol_errors += errors
            total_transmitted_bits += k - np.sum(erasures[:k])  # Count non-erased bits

        # Calculate SER and Bit Rate
        ser = symbol_errors / (k * num_iterations)
        bit_rate = total_transmitted_bits / (k * num_iterations)

        ser_results.append(ser)
        bit_rate_results.append(bit_rate)

    return ser_results, bit_rate_results

# Simulation parameters
n = 49  # Length of codeword, adjusted to be a multiple of d_c
d_v = 4  # Variable node degree for regular LDPC
d_c = 7  # Check node degree for regular LDPC
erasure_thresholds = np.linspace(0.1, 1.0, 10)  # Range of erasure thresholds to test

# Run the simulation
ser_results, bit_rate_results = simulate_ldpc_with_pyldpc(erasure_thresholds, n, d_v, d_c)

# Plotting results
plt.figure(figsize=(14, 6))

# Plot Symbol Error Rate
plt.subplot(1, 2, 1)
plt.plot(erasure_thresholds, ser_results, marker='o')
plt.title("Symbol Error Rate vs. Erasure Threshold")
plt.xlabel("Erasure Threshold")
plt.ylabel("Symbol Error Rate (SER)")
plt.grid()

# Plot Bit Rate
plt.subplot(1, 2, 2)
plt.plot(erasure_thresholds, bit_rate_results, marker='o', color='orange')
plt.title("Bit Rate vs. Erasure Threshold")
plt.xlabel("Erasure Threshold")
plt.ylabel("Bit Rate")
plt.grid()

plt.tight_layout()
plt.show()