import matplotlib.pyplot as plt
import numpy as np
from erasure_channel_encoding import simulate_ldpc_erasure_correction

# Simulation parameters
n = 49  # Length of codeword, adjusted to be a multiple of d_c
d_v = 4  # Variable node degree for regular LDPC
d_c = 7  # Check node degree for regular LDPC
erasure_thresholds = np.linspace(0.1, 1.0, 50)  # Increase points for smooth curves

# Assume `simulate_ldpc_erasure_correction` is your simulation function
ser_results, bit_rate_results = simulate_ldpc_erasure_correction(erasure_thresholds, n, d_v, d_c)

# Plotting results
plt.figure(figsize=(14, 6))

# Plot Symbol Error Rate (SER) with log scale
plt.subplot(1, 2, 1)
plt.plot(erasure_thresholds, ser_results, marker='o', markersize=4)
plt.title("Symbol Error Rate vs. Erasure Threshold")
plt.xlabel("Erasure Threshold")
plt.ylabel("Symbol Error Rate (SER)")
plt.yscale('log')  # Set y-axis to log scale
plt.grid()

# Plot Bit Rate with log scale
plt.subplot(1, 2, 2)
plt.plot(erasure_thresholds, bit_rate_results, marker='o', color='orange', markersize=4)
plt.title("Bit Rate vs. Erasure Threshold")
plt.xlabel("Erasure Threshold")
plt.ylabel("Bit Rate")
plt.yscale('log')  # Set y-axis to log scale
plt.grid()

plt.tight_layout()
plt.show()

# Print final results
for threshold, ser, bit_rate in zip(erasure_thresholds, ser_results, bit_rate_results):
    print(f"Threshold: {threshold}, SER: {ser:.5f}, Bit Rate: {bit_rate:.5f}")
