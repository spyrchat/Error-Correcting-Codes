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

    syndrome = (H @ received_codeword.transpose()) % 2

    # syndrome lookup for single-bit error correction
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
        # if theres an error, correct it
        received_codeword[error_position] ^= 1  # Flip the bit
        print(f"SNR = {snr} dB. Error found at bit position: {error_position}")
        #print(f"Corrected Codeword: {received_codeword}")

    return syndrome, received_codeword

def error_calculation(message, received_codeword):
    errors = np.sum(received_codeword[:4] != message)
    return errors

# simulation parameters
snr_db = np.arange(0, 11)
num_messages = 10**6

errors_per_snr = np.zeros(len(snr_db))

noisy_bits_per_snr = {snr: [] for snr in snr_db}  # form (original_bit, noisy_bit)

# Run the experiment for multiple messages
for _ in range(num_messages):
    original = np.random.randint(0, 2, 4)  # random 4bit message
    print()
    print(f'Original Message: {original}')
    #codeword = hamming_coding(original)
    #print(f'Codeword: {codeword}')
    bpsk_codeword = bpsk_modulation(original)
    print(f'BPSK Modulation: {bpsk_codeword}')
    print('-------------------------------')

    for snr_value in snr_db:
        noisy_codeword = add_awgn(bpsk_codeword, snr_value)
        noisy_bits_per_snr[snr_value].extend(zip(bpsk_codeword,noisy_codeword))
        received_codeword = demodulator(noisy_codeword)
        #syndrome, final_codeword = hamming_decoding(received_codeword, snr_value)
        errors = error_calculation(original, received_codeword)
        errors_per_snr[snr_value] += errors

        if errors != 0:
            print(f'SNR =  {snr_value} dB')
            print(f'Received Message: {received_codeword[:4]}')
            print(f'Errors: {errors}')
            print('-------------------------------')

ber_per_snr = errors_per_snr / (num_messages * 4)
bit_rate_per_snr = 1 - ber_per_snr

print("Summary:")
for snr, errors, ber, bit_rate in zip(snr_db, errors_per_snr, ber_per_snr, bit_rate_per_snr):
    print(f"SNR = {snr} dB: Total Errors = {errors}, BER = {ber:.5f}, Bit Rate = {bit_rate:.5f}")

print(f"Total Errors across all SNRs = {errors_per_snr.sum()}")

# plots
plt.figure(figsize=(12, 6))

# BER vs SNR
plt.subplot(1, 2, 1)
plt.plot(snr_db, ber_per_snr, marker='o', label='BER')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.yscale('log')
plt.title('BER vs SNR')
plt.grid(True)
plt.legend()

# Bit Rate vs SNR
plt.subplot(1, 2, 2)
plt.plot(snr_db, bit_rate_per_snr, marker='s', color='green', label='Bit Rate')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Rate')
plt.title('Bit Rate vs SNR')
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig('BPSK_1.png')

# plot for noisy bits
plt.figure(figsize=(12, 10))

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='g', label='Transmitted Bits', markersize=8, linestyle='None'),
    plt.Line2D([0], [0], marker='x', color='blue', label='Noisy (+1)', markersize=8, linestyle='None'),
    plt.Line2D([0], [0], marker='x', color='red', label='Noisy (-1)', markersize=8, linestyle='None')
]

for idx, snr_value in enumerate(snr_db):
    plt.subplot(4, 3, idx + 1)

    # separate noisy bits based on their original transmitted values
    transmitted_positive = [noisy_bit for original_bit, noisy_bit in noisy_bits_per_snr[snr_value] if
                            original_bit == +1]
    transmitted_negative = [noisy_bit for original_bit, noisy_bit in noisy_bits_per_snr[snr_value] if
                            original_bit == -1]

    # plot original bits
    plt.scatter([-1, 1], [0, 0], marker='o', color='g', s=150, label='Transmitted Bits')

    # plot noisy bits
    plt.scatter(transmitted_positive, [0] * len(transmitted_positive), marker='x', color='blue', label='Noisy (+1)',
                alpha=0.6)
    plt.scatter(transmitted_negative, [0] * len(transmitted_negative), marker='x', color='red', label='Noisy (-1)',
                alpha=0.6)

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.title(f'SNR = {snr_value} dB')
    plt.grid()


plt.figlegend(handles=legend_elements, loc='lower center', ncol=3, frameon=False, fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig('BPSK_2.png')