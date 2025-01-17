from multiprocessing import Pool
from erasure_channel_encoding import simulate_ldpc_erasure_correction
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Non-interactive backend for multiprocessing safe plotting
matplotlib.use('Agg')


# Simulation parameters
n = 49  # Length of codeword, adjusted to be a multiple of d_c
d_v = 4  # Variable node degree for regular LDPC
d_c = 7  # Check node degree for regular LDPC
# Increase points for smooth curves
erasure_thresholds = np.linspace(0.1, 1.0, 50)
snr_values = [10, 7, 5]

# Directory for saving plots
output_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)  # Ensure plots directory exists

# Function to run simulation and save plots


def run_simulation_and_plot(snr):
    # Run the simulation
    ser_results, bit_rate_results = simulate_ldpc_erasure_correction(
        erasure_thresholds, n, d_v, d_c, snr)

    # Avoid log scale issues by ensuring no zero values
    ser_results = np.maximum(ser_results, 1e-10)
    bit_rate_results = np.maximum(bit_rate_results, 1e-10)

    # Plotting results
    plt.figure(figsize=(14, 6))

    # Plot Symbol Error Rate (SER) with log scale
    plt.subplot(1, 2, 1)
    plt.plot(erasure_thresholds, ser_results, marker='o',
             markersize=4, label=f"SNR = {snr}")
    plt.title("Symbol Error Rate vs. Erasure Threshold")
    plt.xlabel("Erasure Threshold")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.legend()
    plt.yscale('log')  # Set y-axis to log scale
    plt.grid()

    # Plot Bit Rate with log scale
    plt.subplot(1, 2, 2)
    plt.plot(erasure_thresholds, bit_rate_results,
             marker='o', markersize=4, label=f"SNR = {snr}")
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
        print(f"SNR = {snr}, Threshold: {threshold:.2f}, SER: {
              ser:.5f}, Bit Rate: {bit_rate:.5f}")


# Main function for multiprocessing
if __name__ == "__main__":
    with Pool() as pool:
        pool.map(run_simulation_and_plot, snr_values)
    print("All simulations completed.")
