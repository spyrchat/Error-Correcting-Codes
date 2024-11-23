import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode, get_message

def simulate_ldpc(erasure_thresholds, n, d_v, d_c, snr_db=10, num_iterations=1000):
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

        for _ in range(num_iterations):
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

            # Calculate errors (exclude erased bits)
            non_erased_indices = ~erasures[:k]
            errors = np.sum(decoded_message[non_erased_indices] != message[non_erased_indices])
            total_errors += errors
            total_non_erased += np.sum(non_erased_indices)

            # Calculate bit rate
            total_transmitted_bits += np.sum(non_erased_indices)

        ser = total_errors / total_non_erased if total_non_erased > 0 else 0
        bit_rate = total_transmitted_bits / (k * num_iterations)

        ser_results.append(ser)
        bit_rate_results.append(bit_rate)

    return ser_results, bit_rate_results
