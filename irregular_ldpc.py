import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode, get_message

def generate_irregular_ldpc(n, d_v, lambda_dist, rho_dist):
    """
    Generate an irregular LDPC parity-check matrix using degree distributions.
    """
    # Calculate d_c
    valid_d_c = [i for i in range(1, n + 1) if n % i == 0 and i > d_v]
    if not valid_d_c:
        raise ValueError(f"No valid d_c found for n={n} and d_v={d_v}. Ensure n % d_c == 0 and d_c > d_v.")
    d_c = max(valid_d_c)

    # Generate LDPC matrix
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    return H, G

# Simulation parameters
n = 480  # Length of codeword
k = 24   # Number of information bits
snr_db = 5  # SNR in dB

# Regular LDPC design
d_v_reg = 3
d_c_reg = max([i for i in range(1, n + 1) if n % i == 0 and i > d_v_reg])
H_reg, G_reg = make_ldpc(n, d_v_reg, d_c_reg, systematic=True, sparse=True)

# Irregular LDPC design
lambda_dist = [0.5, 0.3, 0.2]  # λ(x) = 0.5x^2 + 0.3x^3 + 0.2x^8
rho_dist = [0.6, 0.4]           # ρ(x) = 0.6x^6 + 0.4x^7
H_irr, G_irr = generate_irregular_ldpc(n, k, lambda_dist, rho_dist)

# Simulate performance
def simulate_ldpc_performance(H, G, snr_db, num_iterations=1000):
    """
    Εξομοιώνει την επίδοση LDPC.
    """
    noise_std = np.sqrt(1 / (2 * 10 ** (snr_db / 10)))
    total_errors = 0

    for _ in range(num_iterations):
        message = np.random.randint(0, 2, G.shape[1])
        codeword = encode(G, message, snr_db)

        # Transmit with noise
        transmitted_signal = 2 * codeword - 1
        noise = np.random.normal(0, noise_std, transmitted_signal.shape)
        received_signal = transmitted_signal + noise

        # Decode
        received_signal_scaled = 2 * received_signal / noise_std**2
        decoded_codeword = decode(H, received_signal_scaled, snr=snr_db, maxiter=100)
        decoded_message = get_message(G, decoded_codeword)

        # Count errors
        total_errors += np.sum(decoded_message != message)

    ber = total_errors / (num_iterations * G.shape[1])
    return ber

# Measure BER for regular and irregular LDPC
ber_reg = simulate_ldpc_performance(H_reg, G_reg, snr_db)
ber_irr = simulate_ldpc_performance(H_irr, G_irr, snr_db)

print(f"Regular LDPC BER: {ber_reg:.5e}")
print(f"Irregular LDPC BER: {ber_irr:.5e}")

# Plot BER comparison
plt.bar(["Regular LDPC", "Irregular LDPC"], [ber_reg, ber_irr], color=["blue", "orange"])
plt.title("BER Comparison: Regular vs. Irregular LDPC")
plt.ylabel("Bit Error Rate (BER)")
plt.grid()
plt.show()
