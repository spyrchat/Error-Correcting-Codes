import numpy as np
import matplotlib.pyplot as plt
from decoder_cuda import decode, get_message
from construct_irregular_ldpc import construct_irregular_ldpc
import os
from encoder import encode


def simulate_irregular_ldpc_erasure_correction(
    erasure_thresholds,
    number_of_variable_nodes,
    lambda_dist,
    rho_dist,
    snr_db=10,
    num_iterations=100,
    plot_interval=1000,
    verbose=False
):
    """
    Simulate LDPC encoding, transmission with noise and erasures, and decoding for a single SNR value.

    Parameters:
    - erasure_thresholds: List of thresholds for erasure detection.
    - number_of_variable_nodes: Number of variable nodes in the LDPC code.
    - lambda_dist: Degree distribution for variable nodes.
    - rho_dist: Degree distribution for check nodes.
    - snr_db: Signal-to-Noise Ratio in dB (single value).
    - num_iterations: Number of iterations for simulation.
    - plot_interval: Interval for saving constellation diagrams.
    - verbose: Whether to print detailed debug information.

    Returns:
    - ser_results: Symbol Error Rates (SER) for each erasure threshold.
    - bit_rate_results: Bit Rates for each erasure threshold.
    """
    print(f"Starting LDPC simulation for SNR = {snr_db} dB...")

    # Generate LDPC matrices
    print("Generating LDPC matrices...")
    H, G = construct_irregular_ldpc(
        number_of_variable_nodes, lambda_dist, rho_dist)
    H = H.toarray()
    G = G.toarray()
    k = G.shape[1]  # Number of information bits
    print(f"Generated LDPC matrices: H.shape={H.shape}, G.shape={G.shape}")

    ser_results = []
    bit_rate_results = []

    # SNR and noise parameters
    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))  # Noise standard deviation
    noise_std_sq = noise_std**2  # Precompute noise variance

    # Ensure plots directory exists
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plots will be saved in: {plot_dir}")

    for threshold in erasure_thresholds:
        print(f"\nStarting simulation for threshold {threshold:.2f}...")
        total_errors = 0
        total_non_erased = 0
        total_transmitted_bits = 0

        for iteration in range(num_iterations):
            # Generate random message and encode it
            message = np.random.randint(0, 2, k)
            codeword = encode(G, message, snr=snr_db)

            # Ensure codeword contains binary values (0 and 1)
            codeword = np.round(codeword).astype(int)

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
            received_signal_scaled = 2 * decoder_input / noise_std_sq

            # Decode the received signal
            decoded_codeword = decode(
                H, received_signal_scaled, snr=snr_db, maxiter=100)
            # Extract the first `k` bits (message bits)
            decoded_message_k = decoded_codeword[:k]

            # Debugging: Print the shapes
            print(f"decoded_codeword shape: {decoded_codeword.shape}")
            print(f"decoded_message_k shape: {decoded_message_k.shape}")
            print(f"First 10 decoded_message_k: {decoded_message_k[:10]}")

            # Create a boolean array for non-erased bits (aligned to `k`)
            # Initialize mask for `k` bits
            non_erased_indices_k = np.zeros(k, dtype=bool)
            # Fill up to `k` length
            non_erased_indices_k[:len(erasures)] = ~erasures[:k]

            # Debugging: Print non-erased mask details
            print(f"erasures shape: {erasures.shape}")
            print(f"non_erased_indices_k shape: {non_erased_indices_k.shape}")
            print(
                f"Sum of non-erased bits (k): {np.sum(non_erased_indices_k)}")

            # Slice the message to match `k` length
            valid_message = message[:k]

            # Debugging: Print message details
            print(f"message shape: {message.shape}")
            print(f"valid_message shape: {valid_message.shape}")
            print(f"First 10 valid_message: {valid_message[:10]}")

            # Calculate errors only for non-erased indices (for the first `k` bits)
            errors = np.sum(
                decoded_message_k[non_erased_indices_k] != valid_message[non_erased_indices_k]
            )
            print(f"Iteration {iteration}: Errors={
                  errors}, Non-erased={np.sum(non_erased_indices_k)}")

            total_errors += errors
            total_non_erased += np.sum(non_erased_indices_k)
            total_transmitted_bits += np.sum(non_erased_indices_k)

            # Debugging error statistics
            if verbose:
                print(f"Iteration {iteration}: Errors={
                      errors}, Non-erased={np.sum(non_erased_indices_k)}")

            # Plot constellation diagram at intervals
            if iteration % plot_interval == 0:
                plt.clf()
                plt.scatter(transmitted_signal, np.zeros_like(
                    transmitted_signal), color='blue', label='Transmitted Symbols', s=50)
                plt.scatter(received_signal, np.zeros_like(
                    received_signal), color='green', alpha=0.6, label='Received Symbols', s=50)
                plt.scatter(received_signal[erasures], np.zeros_like(
                    received_signal[erasures]), color='red', alpha=0.8, label='Erased Symbols', s=50)

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
                plt.title(f'Constellation Diagram\n(SNR = {
                          snr_db} dB, Threshold = {threshold:.2f})')
                plt.xlabel('Real Part')
                plt.ylabel('Imaginary Part')
                plt.legend()
                plt.grid()

                # Save the plot
                filename = os.path.join(plot_dir, f"constellation_snr_{
                                        snr_db}_threshold_{threshold:.2f}.png")
                plt.savefig(filename)
                plt.close()

        # SER and Bit Rate
        ser = total_errors / total_non_erased if total_non_erased > 0 else np.nan
        bit_rate = total_transmitted_bits / \
            (k * num_iterations) if total_transmitted_bits > 0 else 0

        print(f"Threshold: {threshold:.2f}, SER: {
              ser:.5f}, Bit Rate: {bit_rate:.5f}")

        ser_results.append(ser)
        bit_rate_results.append(bit_rate)

    print("Simulation completed.")
    return ser_results, bit_rate_results
