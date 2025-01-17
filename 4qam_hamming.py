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

def qam_modulation(codeword):
    if len(codeword) % 2 != 0:
        codeword = np.append(codeword, 0)

    qam_symbols = {
        (0, 0): -1 - 1j,  # Symbol '00'
        (0, 1): -1 + 1j,  # Symbol '01'
        (1, 0): 1 - 1j,   # Symbol '10'
        (1, 1): 1 + 1j    # Symbol '11'
    }

    reshaped_codeword = codeword.reshape((-1, 2))
    qam_codeword = np.array([qam_symbols[tuple(bits)] for bits in reshaped_codeword])

    return qam_codeword

def add_awgn(qam_codeword, snr_db):
    snr = 10.0 ** (snr_db / 10.0)
    std_deviation = np.sqrt(1 / (2 * snr))

    noise_real = np.random.normal(0, std_deviation, qam_codeword.shape)
    noise_imag = np.random.normal(0, std_deviation, qam_codeword.shape)
    noise = noise_real + 1j * noise_imag

    noisy_codeword = qam_codeword + noise
    return noisy_codeword

def demodulator(noisy_codeword):
    # Decode the real and imaginary parts of the QAM symbols back to bits
    received_bits = []
    for symbol in noisy_codeword:
        real_bit = 1 if symbol.real >= 0 else 0
        imag_bit = 1 if symbol.imag >= 0 else 0
        received_bits.extend([real_bit, imag_bit])

    return np.array(received_bits[:7])

def hamming_decoding(received_codeword, snr):
    # Hamming(7,4) parity-check matrix
    H = np.array([[1, 1, 0, 1, 1, 0, 0],
                  [1, 0, 1, 1, 0, 1, 0],
                  [0, 1, 1, 1, 0, 0, 1]])

    # Ensure received_codeword is a 1D array
    received_codeword = np.array(received_codeword).flatten()

    syndrome = (H @ received_codeword) % 2

    # Syndrome lookup for single-bit error correction
    syndrome_lookup = {
        (0, 0, 0): None,
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
        # If there's an error, correct it
        received_codeword[error_position] ^= 1  # flip the bit
        print(f"SNR = {snr} dB. Error found at bit position: {error_position}")

    return syndrome, received_codeword

def error_calculation(original, received_codeword):
    errors = np.sum(received_codeword[:4] != original)
    return errors

# simulation parameters
snr_db = np.arange(0, 11)
num_messages = 10**6

errors_per_snr = np.zeros(len(snr_db))

noisy_bits_per_snr = {snr: [] for snr in snr_db}  # form (original_bit, noisy_bit)

for _ in range(num_messages):
    original = np.random.randint(0, 2, 4)  # random 4bit message
    print()
    print(f'Original Message: {original}')
    codeword = hamming_coding(original)
    print(f'Codeword: {codeword}')
    qam_codeword = qam_modulation(codeword)
    print(f'QAM Modulation: {qam_codeword}')
    print('-------------------------------')

    for snr_value in snr_db:
        noisy_codeword = add_awgn(qam_codeword, snr_value)
        noisy_bits_per_snr[snr_value].extend(zip(qam_codeword,noisy_codeword))
        received_codeword = demodulator(noisy_codeword)
        syndrome, final_codeword = hamming_decoding(received_codeword, snr_value)
        errors = error_calculation(original, final_codeword)
        errors_per_snr[snr_value] += errors

        if errors != 0:
            print(f'SNR =  {snr_value} dB')
            print(f'Received Message: {final_codeword[:4]}')
            print(f'Errors not corrected by Hamming: {errors}')
            print('-------------------------------')

ber_per_snr = errors_per_snr / (num_messages * 4)
bit_rate_per_snr = 4/7 * (1 - ber_per_snr)

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

plt.savefig("QAM_Hamming_1.png")

print('QAM_Hamming_1.png Saved.')

# Plot for noisy bits
plt.figure(figsize=(12, 10))

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='g', label='Transmitted Bits', markersize=8, linestyle='None'),
    plt.Line2D([0], [0], marker='x', color='blue', label='Noisy', markersize=8, linestyle='None')
]

# 4-QAM Transmitted Symbols
transmitted_symbols = [(-1 - 1j), (-1 + 1j), (1 - 1j), (1 + 1j)]

for idx, snr_value in enumerate(snr_db):
    plt.subplot(4, 3, idx + 1)

    # Extract noisy bits regardless of whether they are positive or negative
    noisy_bits = [noisy_bit for original_bit, noisy_bit in noisy_bits_per_snr[snr_value]]

    # Plot original transmitted QAM symbols
    plt.scatter([symbol.real for symbol in transmitted_symbols],
                [symbol.imag for symbol in transmitted_symbols],
                marker='o', color='g', s=150, label='Transmitted Bits')

    # Plot noisy bits (both positive and negative)
    plt.scatter([noisy_bit.real for noisy_bit in noisy_bits],
                [noisy_bit.imag for noisy_bit in noisy_bits],
                marker='x', color='blue', alpha=0.6, label='Noisy')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.title(f'SNR = {snr_value} dB')
    plt.grid()

plt.figlegend(handles=legend_elements, loc='lower center', ncol=2, frameon=False, fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig('QAM_Hamming_2.png')

print('QAM_Hamming_2.png Saved.')
