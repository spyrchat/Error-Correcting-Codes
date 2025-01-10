import numpy as np
from decoder import decode, get_message
from encoder import encode


def test_decoder_with_erasures():

    # Load the saved numpy array
    try:
        H = np.load("H_matrix.npy")
        G = np.load("G_matrix.npy")
    except FileNotFoundError:
        print("Error: One or both of the numpy files 'H_matrix.npy' and 'G_matrix.npy' were not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading numpy files: {e}")
        exit(1)
    k = G.shape[1]  # Number of information bits
    snr_db = 10  # Signal-to-Noise Ratio in dB
    erasure_threshold = 0.29492  # Erasure threshold
    noise_std = 1 / (2 * 10**(snr_db / 10))**0.5

    # Generate a random message
    message = np.random.randint(0, 2, k)
    print(f"Original message (first 10 bits): {message[:10]}...")

    # Encode message
    codeword = encode(G, message, snr=snr_db)

    # Ensure codeword contains binary values (0 and 1)
    codeword = np.round(codeword).astype(int)

    # BPSK modulation and AWGN noise
    transmitted_signal = 2 * codeword - 1
    noise = np.random.normal(0, noise_std, transmitted_signal.shape)
    received_signal = transmitted_signal + noise

    # Erasure condition
    erasures = np.abs(received_signal) < erasure_threshold
    decoder_input = np.copy(received_signal)
    decoder_input[erasures] = 0  # Neutralize erased symbols

    # Scale signal for decoding
    received_signal_scaled = 2 * decoder_input / noise_std**2

    # Decode
    print("Decoding the message...")
    decoded_codeword = decode(
        H, received_signal_scaled, snr=snr_db, maxiter=1000)

    # Extract information bits from the decoded message
    decoded_information_bits = get_message(G, decoded_codeword)

    # Compare original and decoded information bits
    if np.array_equal(message, decoded_information_bits):
        print("Decoder successfully decoded the information bits!")
    else:
        print("Decoder failed to decode the information bits.")
        print(f"Decoded information bits (first 10): {
              decoded_information_bits[:10]}...")
        print(f"Original information bits (first 10): {message[:10]}...")


# Run the test
test_decoder_with_erasures()
