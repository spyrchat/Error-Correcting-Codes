import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode, get_message

def simulate_ldpc_erasure_correction(erasure_thresholds, n, d_v, d_c, snr_db=10, num_iterations=1000, plot_interval=1000):
    """
    Simulate LDPC encoding, transmission with noise and erasures, and decoding.

    Returns:
    - ser_results: Symbol Error Rates (SER) for each erasure threshold.
    - bit_rate_results: Bit Rates for each erasure threshold.
    """
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]  # Number of information bits

    ser_results = []
    bit_rate_results = []

    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))  # Noise standard deviation

    for threshold in erasure_thresholds:
        total_errors = 0
        total_non_erased = 0
        total_transmitted_bits = 0

        for iteration in range(num_iterations):
            # Generate random message and encode it
            message = np.random.randint(0, 2, k)
            codeword = encode(G, message, snr_db)

            # BPSK modulation
            transmitted_signal = 2 * codeword - 1

            # Add AWGN
            noise = np.random.normal(0, noise_std, transmitted_signal.shape)
            received_signal = transmitted_signal + noise

            # Apply erasure threshold
            erasures = np.abs(received_signal) < threshold
            received_signal[erasures] = 0  # Neutral value for erased symbols

            # Scale signal for decoding
            received_signal_scaled = 2 * received_signal / noise_std**2
            decoded_codeword = decode(H, received_signal_scaled, snr=snr_db, maxiter=100)
            decoded_message = get_message(G, decoded_codeword)

            # Print transmitted and decoded messages
            print(f"Iteration {iteration + 1}, Threshold {threshold}")
            print(f"Transmitted Message: {message}")
            print(f"Decoded Message:    {decoded_message}")

            # Calculate errors (exclude erased bits)
            non_erased_indices = ~erasures[:k]
            errors = np.sum(decoded_message[non_erased_indices] != message[non_erased_indices])
            total_errors += errors
            total_non_erased += np.sum(non_erased_indices)

            # Calculate bit rate
            total_transmitted_bits += np.sum(non_erased_indices)

            # Plot constellation diagram at intervals
            if iteration % plot_interval == 0:
                plt.clf()
                plt.scatter(transmitted_signal, np.zeros_like(transmitted_signal), color='blue', label='Transmitted Symbols', s=100)
                plt.scatter(received_signal, np.zeros_like(received_signal), color='green', alpha=0.6, label='Received Symbols', s=100)
                plt.scatter(received_signal[erasures], np.zeros_like(received_signal[erasures]), color='red', alpha=0.8, label='Erased Symbols', s=100)
                plt.gca().add_patch(
                    plt.Rectangle(
                        (-threshold, -0.05),
                        2 * threshold,
                        0.1,
                        color='orange',
                        alpha=0.3,
                        label=f'Erasure Threshold = {threshold}'
                    )
                )
                plt.axhline(0, color='black', linewidth=1)
                plt.axvline(0, color='black', linewidth=1)
                plt.xlim(-2, 2)
                plt.ylim(-0.5, 0.5)
                plt.title(f'Constellation Diagram (Threshold {threshold}, Iteration {iteration + 1})')
                plt.xlabel('Real Part')
                plt.ylabel('Imaginary Part')
                plt.legend()
                plt.grid()
                plt.pause(0.001)

        ser = total_errors / total_non_erased if total_non_erased > 0 else 0
        bit_rate = total_transmitted_bits / (k * num_iterations)

        ser_results.append(ser)
        bit_rate_results.append(bit_rate)

    plt.show()
    return ser_results, bit_rate_results