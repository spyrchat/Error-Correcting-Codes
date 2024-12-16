import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode, get_message
import os

def simulate_ldpc_erasure_correction(erasure_thresholds, n, d_v, d_c, snr_db=10, num_iterations=1000, plot_interval=1000, verbose=False):
    """
    Simulate LDPC encoding, transmission with noise and erasures, and decoding.

    Returns:
    - ser_results: Symbol Error Rates (SER) for each erasure threshold.
    - bit_rate_results: Bit Rates for each erasure threshold.
    """
    # Generate LDPC matrices
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]  # Number of information bits
    print(f"Generated LDPC matrices: H.shape={H.shape}, G.shape={G.shape}")

    ser_results = []
    bit_rate_results = []

    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))  # Noise standard deviation

    # Ensure plots directory exists
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for threshold in erasure_thresholds:
        print(f"Starting simulation for threshold {threshold:.2f} (SNR = {snr_db} dB)")
        total_errors = 0
        total_non_erased = 0
        total_transmitted_bits = 0

        corrected_from_erasures_count = 0
        corrected_from_noise_count = 0
        uncorrected_symbols_count = 0

        for iteration in range(num_iterations):
            # Generate random message and encode it
            message = np.random.randint(0, 2, k)
            codeword = encode(G, message,snr_db)

            # BPSK modulation and AWGN noise
            transmitted_signal = 2 * codeword - 1
            noise = np.random.normal(0, noise_std, transmitted_signal.shape)
            received_signal = transmitted_signal + noise

            # Erasure condition
            erasures = np.abs(received_signal) < threshold

            # Prepare decoder input
            decoder_input = np.copy(received_signal)
            decoder_input[erasures] = 0  # Neutralize erased symbols

            # Scale signal for decoding
            received_signal_scaled = 2 * decoder_input / noise_std**2

            # Decode
            decoded_codeword = decode(H, received_signal_scaled, snr=snr_db, maxiter=100)
            decoded_message = get_message(G, decoded_codeword)

            # Corrected symbols
            corrected_from_erasures = erasures[:k] & (decoded_message == message)
            corrected_from_noise = ~erasures[:k] & (decoded_message == message)

            # Uncorrected symbols
            uncorrected_symbols = decoded_message != message

            # Update counts
            corrected_from_erasures_count += np.sum(corrected_from_erasures)
            corrected_from_noise_count += np.sum(corrected_from_noise)
            uncorrected_symbols_count += np.sum(uncorrected_symbols)

            # Calculate errors: Ignore erased bits
            non_erased_indices = ~erasures[:k]
            errors = np.sum(decoded_message[non_erased_indices] != message[non_erased_indices])
            total_errors += errors
            total_non_erased += np.sum(non_erased_indices)
            total_transmitted_bits += np.sum(non_erased_indices)

            # Plot constellation diagram at intervals
            if iteration % plot_interval == 0:
                plt.clf()
                plt.scatter(transmitted_signal, np.zeros_like(transmitted_signal), color='blue', label='Transmitted Symbols', s=50)
                plt.scatter(received_signal, np.zeros_like(received_signal), color='green', alpha=0.6, label='Received Symbols', s=50)
                plt.scatter(received_signal[erasures], np.zeros_like(received_signal[erasures]), color='red', alpha=0.8, label='Erased Symbols', s=50)
                plt.scatter(received_signal[np.where(corrected_from_erasures)], np.zeros_like(received_signal[np.where(corrected_from_erasures)]),
                            color='purple', alpha=0.8, label='Corrected from Erasures', s=100)
                plt.scatter(received_signal[np.where(corrected_from_noise)], np.zeros_like(received_signal[np.where(corrected_from_noise)]),
                            color='cyan', alpha=0.8, label='Corrected from Noise', s=100)

                plt.gca().add_patch(
                    plt.Rectangle(
                        (-threshold, -0.05),
                        2 * threshold,
                        0.1,
                        color='orange',
                        alpha=0.3,
                        label=f'Erasure Threshold = {threshold:.2f}'
                    )
                )
                plt.axhline(0, color='black', linewidth=1)
                plt.axvline(0, color='black', linewidth=1)
                plt.xlim(-2, 2)
                plt.ylim(-0.5, 0.5)
                plt.title(f'Constellation Diagram\n(SNR = {snr_db} dB, Threshold = {threshold:.2f})')
                plt.xlabel('Real Part')
                plt.ylabel('Imaginary Part')
                plt.legend()
                plt.grid()

                # Save the plot
                filename = os.path.join(plot_dir, f"constellation_snr_{snr_db}_threshold_{threshold:.2f}.png")
                plt.savefig(filename)
                plt.close()

        # Compute SER and bit rate
        ser = total_errors / total_non_erased if total_non_erased > 0 else np.nan
        bit_rate = total_transmitted_bits / (k * num_iterations) if total_transmitted_bits > 0 else 0

        print(f"Threshold: {threshold:.2f}, SER: {ser:.5f}, Bit Rate: {bit_rate:.5f}")
        print(f"Corrected from erasures: {corrected_from_erasures_count}")
        print(f"Corrected from noise: {corrected_from_noise_count}")
        print(f"Uncorrected symbols: {uncorrected_symbols_count}")

        ser_results.append(ser)
        bit_rate_results.append(bit_rate)

    return ser_results, bit_rate_results
