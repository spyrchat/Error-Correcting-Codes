import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multiprocessing safe plotting

import matplotlib.pyplot as plt
import numpy as np
import os
from erasure_channel_encoding_irregular import simulate_irregular_ldpc_erasure_correction
from multiprocessing import Pool

Lambda = np.array([
    0.3442, 1.715e-06, 1.441e-06, 1.135e-06, 7.939e-07, 4.122e-07, 0, 0, 0, 0, 
    0.03145, 0.21, 0, 0.1383, 0.276
])
Lambda /= Lambda.sum()  # Normalize to sum to 1

# Provided Rho(x) coefficients (length 15, highest order first)
rho = np.array([
    0.50913838, 0.49086162, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
])
rho /= rho.sum()  # Normalize to sum to 1

# Design parameters
design_rate = 0.744
Lambda_prime = np.dot(np.arange(1, len(Lambda) + 1), Lambda[::-1])  # Reverse order
rho_prime = np.dot(np.arange(1, len(rho) + 1), rho[::-1])  # Reverse order
n = int(np.ceil((rho_prime / (1 - design_rate)) ** 2))
erasure_thresholds = np.linspace(0.1, 1.0, 50)  # Increase points for smooth curves
snr_values = [3, 5, 10]

# Directory for saving plots
output_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)  # Ensure plots directory exists

# Function to run simulation and save plots
def run_simulation_and_plot(snr):
    # Run the simulation
    ser_results, bit_rate_results = simulate_irregular_ldpc_erasure_correction(erasure_thresholds, n, Lambda, rho, snr_db=snr)
    
    # Avoid log scale issues by ensuring no zero values
    ser_results = np.maximum(ser_results, 1e-10)
    bit_rate_results = np.maximum(bit_rate_results, 1e-10)

    # Plotting results
    plt.figure(figsize=(14, 6))

    # Plot Symbol Error Rate (SER) with log scale
    plt.subplot(1, 2, 1)
    plt.plot(erasure_thresholds, ser_results, marker='o', markersize=4, label=f"SNR = {snr}")
    plt.title("Symbol Error Rate vs. Erasure Threshold")
    plt.xlabel("Erasure Threshold")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.legend()
    plt.yscale('log')  # Set y-axis to log scale
    plt.grid()

    # Plot Bit Rate with log scale
    plt.subplot(1, 2, 2)
    plt.plot(erasure_thresholds, bit_rate_results, marker='o', markersize=4, label=f"SNR = {snr}")
    plt.title("Bit Rate vs. Erasure Threshold")
    plt.xlabel("Erasure Threshold")
    plt.ylabel("Bit Rate")
    plt.legend()
    plt.yscale('log')  # Set y-axis to log scale
    plt.grid()

    plt.tight_layout()

    # Save the plot
    filename = os.path.join(plot_dir, f"results_snr_{snr}.png")
    plt.savefig(filename)
    print(f"Saved plot for SNR = {snr} at: {filename}")
    plt.close()  # Close the figure

    # Print final results
    for threshold, ser, bit_rate in zip(erasure_thresholds, ser_results, bit_rate_results):
        print(f"SNR = {snr}, Threshold: {threshold:.2f}, SER: {ser:.5f}, Bit Rate: {bit_rate:.5f}")

# Main function for multiprocessing
if __name__ == "__main__":
    with Pool() as pool:
        pool.map(run_simulation_and_plot, snr_values)

    print("All simulations completed.")
