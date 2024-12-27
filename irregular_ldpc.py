import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multiprocessing safe plotting

import matplotlib.pyplot as plt
import numpy as np
import os
from make_ldpc import make_ldpc_irregular
from pyldpc import make_ldpc

def simulate_ldpc_erasure_correction(erasure_thresholds, n, regular=True, lambda_dist=None, rho_dist=None, snr=10):
    """
    Simulates the LDPC erasure correction for given parameters.

    Parameters:
    erasure_thresholds (numpy.ndarray): Array of erasure thresholds.
    n (int): Length of codeword.
    regular (bool): If True, generates regular LDPC codes; otherwise, irregular.
    lambda_dist (list): Variable node degree distribution for irregular LDPC.
    rho_dist (list): Check node degree distribution for irregular LDPC.
    snr (float): Signal-to-noise ratio.

    Returns:
    tuple: Symbol error rate (SER) and bit rate results.
    """
    if regular:
        # Generate regular LDPC matrices
        d_v = 4
        d_c = 7
        H, G = make_ldpc(n, d_v=d_v, d_c=d_c, systematic=True, sparse=True)
    else:
        # Generate irregular LDPC matrices
        H, G = make_ldpc_irregular(n, lambda_dist=lambda_dist, rho_dist=rho_dist, systematic=True, sparse=True)

    k = G.shape[1]  # Number of information bits

    ser_results = []
    bit_rate_results = []

    snr_linear = 10 ** (snr / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))

    for threshold in erasure_thresholds:
        total_errors = 0
        total_non_erased = 0
        total_transmitted_bits = 0

        for _ in range(1000):  # Simulating 1000 iterations
            message = np.random.randint(0, 2, k)
            codeword = np.dot(message, G.T) % 2

            # Modulate and add noise
            transmitted_signal = 2 * codeword - 1
            noise = np.random.normal(0, noise_std, transmitted_signal.shape)
            received_signal = transmitted_signal + noise

            # Erasure condition
            erasures = np.abs(received_signal) < threshold
            decoder_input = np.copy(received_signal)
            decoder_input[erasures] = 0

            # Decode (placeholder for decoding function)
            decoded_codeword = np.copy(codeword)  # Replace with actual decoding logic
            errors = np.sum(decoded_codeword[:k] != message[:k])

            total_errors += errors
            total_non_erased += k - np.sum(erasures)
            total_transmitted_bits += k

        # Compute SER and bit rate
        ser = total_errors / total_non_erased if total_non_erased > 0 else np.nan
        bit_rate = total_transmitted_bits / (n * 1000) if total_transmitted_bits > 0 else 0

        ser_results.append(ser)  # Append one value per threshold
        bit_rate_results.append(bit_rate)

    return ser_results, bit_rate_results

# Simulation parameters
n = 49  # Length of codeword, adjusted to be a multiple of d_c

lambda_dist_designed = [0.3442, 0, 0.276, 0, 0.1383, 0.21, 0, 0, 0, 0.03145, 0, 0, 0, 0, 1.715e-6, 0.3442]  # Designed lambda distribution
rho_dist_designed = [0.5, 0.5]  # Example rho distribution for designed LDPC

erasure_thresholds = np.linspace(0.1, 1.0, 50)  # Increase points for smooth curves
snr = 10  # Single SNR value

# Directory for saving plots
output_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)  # Ensure plots directory exists

def run_simulation_and_plot():
    """
    Simulates and plots performance of regular and irregular LDPC codes.
    """
    # Regular LDPC
    ser_regular, bit_rate_regular = simulate_ldpc_erasure_correction(erasure_thresholds, n, regular=True, snr=snr)

    # Designed Irregular LDPC
    ser_designed, bit_rate_designed = simulate_ldpc_erasure_correction(erasure_thresholds, n, regular=False, lambda_dist=lambda_dist_designed, rho_dist=rho_dist_designed, snr=snr)

    plt.figure(figsize=(14, 6))

    # Left subplot: SER comparison
    plt.subplot(1, 2, 1)
    plt.plot(erasure_thresholds, ser_regular, label='Regular SER', marker='o', markersize=4)
    plt.plot(erasure_thresholds, ser_designed, label='Designed SER', marker='s', markersize=4)
    plt.title("Symbol Error Rate (SER) vs Erasure Threshold")
    plt.xlabel("Erasure Threshold")
    plt.ylabel("Symbol Error Rate")
    plt.legend()
    plt.grid()

    # Right subplot: Code Rate comparison
    plt.subplot(1, 2, 2)
    plt.plot(erasure_thresholds, bit_rate_regular, label='Regular Code Rate', marker='o', markersize=4)
    plt.plot(erasure_thresholds, bit_rate_designed, label='Designed Code Rate', marker='s', markersize=4)
    plt.title("Code Rate vs Erasure Threshold")
    plt.xlabel("Erasure Threshold")
    plt.ylabel("Code Rate")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "comparison_SER_CodeRate.png"))
    plt.close()

    print(f"Final Regular SER: {ser_regular[-1]:.4f}")
    print(f"Final Designed SER: {ser_designed[-1]:.4f}")
    print(f"Final Regular Code Rate: {bit_rate_regular[-1]:.4f}")
    print(f"Final Designed Code Rate: {bit_rate_designed[-1]:.4f}")

if __name__ == "__main__":
    run_simulation_and_plot()
    print("Simulation completed.")