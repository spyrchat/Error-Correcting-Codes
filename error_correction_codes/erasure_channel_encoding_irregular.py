import numpy as np
import matplotlib.pyplot as plt
from decoder_cuda import decode, get_message
from construct_irregular_ldpc import construct_irregular_ldpc
import os
from encoder import encode

def simulate_irregular_ldpc_erasure_correction(erasure_thresholds, number_of_variable_nodes, lambda_dist, rho_dist, snr_db=10, num_iterations=100, plot_interval=1000, verbose=False):
    """
    Simulate LDPC encoding, transmission with noise and erasures, and decoding.

    Returns:
    - ser_results: Symbol Error Rates (SER) for each erasure threshold.
    - bit_rate_results: Bit Rates for each erasure threshold.
    """
    print("Starting LDPC simulation...")

    # Generate LDPC matrices
    print("Generating LDPC matrices...")
    H, G = construct_irregular_ldpc(number_of_variable_nodes, lambda_dist, rho_dist)
    H = H.toarray()
    G = G.toarray()
    k = G.shape[1]  # Number of information bits
    print(f"Generated LDPC matrices: H.shape={H.shape}, G.shape={G.shape}")

    ser_results = []
    bit_rate_results = []

    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))  # Noise standard deviation
    noise_std_sq = noise_std**2  # Precompute noise variance

    # Ensure plots directory exists
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plots will be saved in: {plot_dir}")

    for threshold in erasure_thresholds:
        print(f"\nStarting simulation for threshold {threshold:.2f} (SNR = {snr_db} dB)")
        total_errors = 0
        total_non_erased = 0
        total_transmitted_bits = 0

        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}...")

            # Generate random message and encode it
            message = np.random.randint(0, 2, k)
            codeword = encode(G, message, snr=snr_db)
            print(f"Generated message: {message[:10]}... (first 10 bits)")
            print(f"Encoded codeword: {codeword[:10]}... (first 10 bits)")

            # Ensure codeword contains binary values (0 and 1)
            codeword = np.round(codeword).astype(int)

            # BPSK modulation and AWGN noise
            transmitted_signal = 2 * codeword - 1
            noise = np.random.normal(0, noise_std, transmitted_signal.shape)
            received_signal = transmitted_signal + noise
            print(f"Transmitted signal: {transmitted_signal[:10]}... (first 10 values)")
            print(f"Received signal (with noise): {received_signal[:10]}... (first 10 values)")

            # Erasure condition
            erasures = np.abs(received_signal) < threshold
            print(f"Erasure mask: {erasures[:10]}... (first 10 values)")

            # Prepare decoder input
            decoder_input = np.copy(received_signal)
            decoder_input[erasures] = 0  # Neutralize erased symbols
            print(f"Decoder input: {decoder_input[:10]}... (first 10 values)")

            # Scale signal for decoding
            received_signal_scaled = 2 * decoder_input / noise_std_sq
            print(f"Scaled received signal: {received_signal_scaled[:10]}... (first 10 values)")

            # Decode
            print("Starting decoding...")
            decoded_codeword = decode(H, received_signal_scaled, snr=snr_db, maxiter=100)
            print(f"Decoded codeword: {decoded_codeword[:10]}... (first 10 bits)")

            # Extract the first k bits (message bits) from the decoded codeword
            message_bits = decoded_codeword[:k]  # Ensure it matches message length
            print(f"Message bits (from decoded codeword): {message_bits[:10]}... (first 10 bits)")

            # Ensure the correct portion of decoded_codeword is passed to get_message
            if decoded_codeword.shape[0] != G.shape[1]:
                raise ValueError(
                    f"Dimension mismatch: decoded_codeword has {decoded_codeword.shape[0]} elements, "
                    f"but G expects {G.shape[1]} columns."
            )

            # Use only the relevant portion of decoded_codeword
            decoded_message = get_message(G.T, decoded_codeword[:G.shape[1]])
            print(f"Decoded message: {decoded_message[:10]}... (first 10 bits)")

            # Calculate errors: Ignore erased bits
            non_erased_indices = ~erasures[:k]
            errors = np.sum(decoded_message[non_erased_indices] != message[non_erased_indices])
            total_errors += errors
            total_non_erased += np.sum(non_erased_indices)
            total_transmitted_bits += np.sum(non_erased_indices)
            print(f"Errors this iteration: {errors}, Total errors so far: {total_errors}")

            # Plot constellation diagram at intervals
            if iteration % plot_interval == 0:
                print(f"Plotting constellation diagram at iteration {iteration}...")
                plt.clf()
                plt.scatter(transmitted_signal, np.zeros_like(transmitted_signal), color='blue', label='Transmitted Symbols', s=50)
                plt.scatter(received_signal, np.zeros_like(received_signal), color='green', alpha=0.6, label='Received Symbols', s=50)
                plt.scatter(received_signal[erasures], np.zeros_like(received_signal[erasures]), color='red', alpha=0.8, label='Erased Symbols', s=50)

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

        ser_results.append(ser)
        bit_rate_results.append(bit_rate)

    print("Simulation completed.")
    return ser_results, bit_rate_results

