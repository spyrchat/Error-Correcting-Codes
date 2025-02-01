import sys
from erasure_channel_encoding_irregular import simulate_irregular_ldpc_erasure_correction
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Non-interactive backend for multiprocessing safe plotting
matplotlib.use('Agg')


# Import the irregular LDPC simulation function directly
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def run_simulation_and_plot(snr_values, H, G):
    erasure_thresholds = np.linspace(0.1, 1.0, 50)
    # Directory for saving plots
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)  # Ensure plots directory exists

    # Handle single SNR value by converting it to a list
    if isinstance(snr_values, (int, float)):
        snr_values = [snr_values]

    # Run the simulation
    for snr in snr_values:
        ser_results, bit_rate_results = simulate_irregular_ldpc_erasure_correction(
            H, G, erasure_thresholds, snr_db=snr)

        # Avoid log scale issues by ensuring no zero values
        ser_results = np.maximum(ser_results, 1e-10)
        bit_rate_results = np.maximum(bit_rate_results, 1e-10)

        # Find the minimum SER and corresponding threshold
        min_ser = np.min(ser_results)
        min_ser_index = np.argmin(ser_results)
        min_ser_threshold = erasure_thresholds[min_ser_index]

        # Plotting results
        plt.figure(figsize=(14, 6))

        # Plot Symbol Error Rate (SER) with log scale
        plt.subplot(1, 2, 1)
        plt.plot(erasure_thresholds, ser_results, marker='o',
                 markersize=4, label=f"SNR = {snr}")
        # Mark the minimum SER point
        plt.scatter(min_ser_threshold, min_ser, color='red', zorder=5,
                    label=f"Min SER: {min_ser:.5e} at {min_ser_threshold:.2f}")
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
    return ser_results, bit_rate_results


# Main function for multiprocessing
if __name__ == "__main__":
    try:
        H = np.load("H_matrix.npy")
        G = np.load("G_matrix.npy")
    except FileNotFoundError:
        print("Error: One or both of the numpy files 'H_matrix.npy' and 'G_matrix.npy' were not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading numpy files: {e}")
        exit(1)
    snr_values = [7]
    run_simulation_and_plot(snr_values, H, G)
    print("All simulations completed.")
