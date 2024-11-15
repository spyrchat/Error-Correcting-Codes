from dbm import error
from idlelib.iomenu import errors

import numpy as np
import matplotlib.pyplot as plt

def hamming_coding(original):
    # Hamming(7,4) generator matrix
    G = np.array([[1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 0, 0, 1, 0, 1],
                  [0, 0, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1]])

    codeword = (original @ G) % 2
    return codeword

def bpsk_modulation(codeword):
    return np.array([1 if bit == 1 else -1 for bit in codeword])

def add_awgn(bpsk_codeword, snr_db):
    snr = 10.0 ** (snr_db / 10.0)
    std_deviation = np.sqrt(1 / (2 * snr))
    noise = np.random.normal(0, std_deviation, bpsk_codeword.shape)
    noisy_codeword = bpsk_codeword + noise
    return noisy_codeword

def demodulator(noisy_codeword):
    received_codeword = np.array([1 if bit >= 0 else 0 for bit in noisy_codeword])
    return received_codeword

def hamming_decoding(received_codeword, snr):
    # Hamming(7,4) parity-check matrix
    H = np.array([[1, 1, 0, 1, 1, 0, 0],
                  [1, 0, 1, 1, 0, 1, 0],
                  [0, 1, 1, 1, 0, 0, 1]])

    # Calculate the syndrome
    syndrome = (H @ received_codeword.transpose()) % 2

    # Syndrome lookup table for single-bit error correction
    syndrome_lookup = {
        (0, 0, 0): None,  # No error
        (1, 1, 0): 0,
        (1, 0, 1): 1,
        (0, 1, 1): 2,
        (1, 1, 1): 3,
        (1, 0, 0): 4,
        (0, 1, 0): 5,
        (0, 0, 1): 6
    }

    syndrome_tuple = tuple(syndrome)

    error_position = syndrome_lookup.get(syndrome_tuple)

    if error_position is not None:
        # Correct the error in received codeword
        received_codeword[error_position] ^= 1  # Flip the bit
        print(f"SNR = {snr} dB. Error found at bit position: {error_position}")
        #print(f"Corrected Codeword: {received_codeword}")

    return syndrome, received_codeword

def error_calculation(message, received_codeword):
    errors = np.sum(received_codeword[:4] != message)
    return errors

# Experiment settings
snr_db = np.arange(0, 11)  # SNR values from 0 to 10 dB
num_messages = 10000  # Number of different messages to test

errors_per_snr = np.zeros(len(snr_db))

# Run the experiment for multiple messages
for _ in range(num_messages):
    original = np.random.randint(0, 2, 4)  # Generate a random 4-bit message
    print()
    print(f'Original Message: {original}')
    codeword = hamming_coding(original)
    print(f'Codeword: {codeword}')
    bpsk_codeword = bpsk_modulation(codeword)
    print(f'BPSK Modulation: {bpsk_codeword}')
    print('-------------------------------')

    for snr_value in snr_db:
        noisy_codeword = add_awgn(bpsk_codeword, snr_value)
        received_codeword = demodulator(noisy_codeword)
        syndrome, final_codeword = hamming_decoding(received_codeword, snr_value)
        errors = error_calculation(original, final_codeword)
        errors_per_snr[snr_value] += errors

        if errors != 0:
            print(f'SNR(dB): {snr_value}')
            print(f'Received Message: {final_codeword[:4]}')
            print(f'Errors not corrected by Hamming: {errors}')
            print('-------------------------------')

ber_per_snr = errors_per_snr / (num_messages * 4)

print("Summary")

# Print the number of errors for each SNR value
for snr, errors in zip(snr_db, errors_per_snr):
    print(f"SNR = {snr} dB: Total Errors = {errors}, BER = {errors / (num_messages * 4)}")

print(f"Total Errors across all SNRs = {errors_per_snr.sum()}")

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(snr_db, ber_per_snr, marker='o', label='BER per SNR')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
