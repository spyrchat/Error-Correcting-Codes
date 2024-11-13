import matplotlib.pyplot as plt
import numpy as np
from erasure_channel_encoding import simulate_ldpc_with_manual_decoding

# Simulation parameters
erasure_thresholds = np.linspace(0.1, 1.0, 10)  # Range of erasure thresholds to test

# Run the simulation
ser_results, bit_rate_results = simulate_ldpc_with_manual_decoding(erasure_thresholds)

# Plotting results
plt.figure(figsize=(14, 6))

# Plot Symbol Error Rate
plt.subplot(1, 2, 1)
plt.plot(erasure_thresholds, ser_results, marker='o')
plt.title("Symbol Error Rate vs. Erasure Threshold")
plt.xlabel("Erasure Threshold")
plt.ylabel("Symbol Error Rate (SER)")
plt.grid()

# Plot Bit Rate
plt.subplot(1, 2, 2)
plt.plot(erasure_thresholds, bit_rate_results, marker='o', color='orange')
plt.title("Bit Rate vs. Erasure Threshold")
plt.xlabel("Erasure Threshold")
plt.ylabel("Bit Rate")
plt.grid()

plt.tight_layout()
plt.show()
