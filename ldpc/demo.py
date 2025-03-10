import numpy as np
from construct_irregular_ldpc import PEG, coding_matrix, validate_ldpc
import matplotlib.pyplot as plt
from ldpc_optimization import c_avg_to_rho, find_best_rate
from simulation_ex3 import run_simulation_and_plot
from simulation_ex2 import run_simulation_and_plot as run_simulation_and_plot_regular
import os


def optimize_ldpc(target_rate, dv_max, dc_max, epsilon_start=0.5, delta_epsilon=0.5, t_delta=0.001):
    """
    Optimizes the LDPC code by finding the best degree distributions and parameters.

    Parameters:
    target_rate (float): Desired design rate.
    dv_max (int): Maximum variable node degree.
    dc_max (int): Maximum check node degree.
    epsilon_start (float): Initial channel parameter.
    delta_epsilon (float): Initial step for epsilon adjustments.
    t_delta (float): Convergence threshold for epsilon.

    Returns:
    dict: Contains the optimized design rate, lambda, rho, c_avg, and epsilon.
    """
    epsilon = epsilon_start
    best_solution = None
    best_threshold = None

    while delta_epsilon >= t_delta:
        print(f"Running optimization for epsilon = {epsilon:.5f}...")
        rate, Lambda, c_avg = find_best_rate(epsilon, dv_max, dc_max)
        print(f"Current Rate: {rate:.3f}, Lambda: {
              Lambda}, c_avg: {c_avg:.3f}")

        if rate > target_rate:
            epsilon += delta_epsilon / 2
        else:
            epsilon -= delta_epsilon / 2

        if best_solution is None or rate > best_threshold:
            best_solution = {
                "rate": rate,
                "lambda": Lambda,
                "rho": c_avg_to_rho(c_avg),
                "c_avg": c_avg,
                "epsilon": epsilon,
            }
            best_threshold = rate
            print(f"New best solution found: Rate = {
                  rate:.3f}, Epsilon = {epsilon:.5f}")

        delta_epsilon /= 2

    print("\nOptimization Complete.")
    return best_solution


def construct_and_plot_ldpc(optimal_params):
    """
    Constructs the LDPC matrices and plots the degree distributions.

    Parameters:
    optimal_params (dict): Contains optimized lambda, rho, and other parameters.

    Returns:
    tuple: Parity-check matrix (H), generator matrix (G), and codeword length (n).
    """
    # Extract parameters
    Lambda = optimal_params["lambda"]
    Rho = optimal_params["rho"]
    design_rate = optimal_params["rate"]

    print("Constructing LDPC matrices...")
    Lambda_prime = np.dot(np.arange(1, len(Lambda) + 1), np.flip(Lambda))
    Rho_prime = np.dot(np.arange(1, len(Rho) + 1), np.flip(Rho))

    N = int(np.ceil((Rho_prime / (1 - design_rate)) ** 2))
    M = int(np.floor((Lambda_prime / Rho_prime) * N))

    degree_sequence = np.random.choice(
        np.arange(1, len(Lambda) + 1), size=N, p=Lambda / np.sum(Lambda)
    )

    peg_instance = PEG(
        nvar=N, nchk=M, degree_sequence=degree_sequence)
    peg_instance.progressive_edge_growth()

    print(f"Generated Codeword Length (n): {N}")
    H = peg_instance.H
    G = coding_matrix(H)

    print(f"Generated Parity-Check Matrix H (shape: {H.shape})")
    print(f"Generated Generator Matrix G (shape: {G.shape})")
    validate_ldpc(H, np.transpose(G))

    # Plot the degree distributions
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(1, len(Lambda) + 1), Lambda, color="blue", alpha=0.7)
    plt.title("Variable Node Degree Distribution (Lambda)")
    plt.xlabel("Degree")
    plt.ylabel("Probability")

    plt.subplot(1, 2, 2)
    plt.bar(np.arange(1, len(Rho) + 1), Rho, color="green", alpha=0.7)
    plt.title("Check Node Degree Distribution (Rho)")
    plt.xlabel("Degree")
    plt.ylabel("Probability")

    plt.tight_layout()
    plt.show()

    return H, G, N


if __name__ == "__main__":
    # User-defined parameters
    target_rate = 0.5
    dv_max = 6
    dc_max = 12
    erasure_threshold = np.linspace(0.1, 1.0, 50)

    print("Optimizing LDPC parameters...")
    # Step 1: Optimize LDPC parameters
    optimal_params = optimize_ldpc(target_rate, dv_max, dc_max)

    print("\nOptimized Parameters:")
    print(f"Design Rate: {optimal_params['rate']:.3f}")
    print(f"Lambda Polynomial: {np.poly1d(
        optimal_params['lambda'], variable='Z')}")
    print(f"Rho Polynomial: {np.poly1d(optimal_params['rho'], variable='Z')}")
    print(f"Average Check Node Degree: {optimal_params['c_avg']:.3f}")
    print(f"Epsilon: {optimal_params['epsilon']:.5f}")

    # Step 2: Construct LDPC matrices and plot degree distributions
    H, G, N = construct_and_plot_ldpc(optimal_params)

    # Optional: Save the matrices for reuse
    np.save("H_matrix.npy", H)
    np.save("G_matrix.npy", G)
    print("Parity-Check Matrix (H) and Generator Matrix (G) saved as .npy files.")

    # Step 3: Run simulation
    snr_values = [10]
    print("Running simulation and plotting results...")
    ser_irregular, ber_irregular = run_simulation_and_plot(snr_values, H, G)
    print("Simulation completed and plots saved.")

    # Step 4: Combine the plots
    ser_regular, ber_regular = run_simulation_and_plot_regular(snr_values[0])

    # Find minimum values and their indices
    min_ser_regular = np.min(ser_regular)
    min_ser_irregular = np.min(ser_irregular)
    min_ser_regular_idx = np.argmin(ser_regular)
    min_ser_irregular_idx = np.argmin(ser_irregular)

    min_ber_regular = np.min(ber_regular)
    min_ber_irregular = np.min(ber_irregular)
    min_ber_regular_idx = np.argmin(ber_regular)
    min_ber_irregular_idx = np.argmin(ber_irregular)

    # Create the plots
    plt.figure(figsize=(12, 6))

    # Plot Symbol Error Rate (SER)
    plt.subplot(1, 2, 1)
    plt.plot(erasure_threshold, ser_regular,
             label="Regular LDPC", marker='o', color='blue')
    plt.plot(erasure_threshold, ser_irregular,
             label="Irregular LDPC", marker='s', color='red')
    plt.scatter(erasure_threshold[min_ser_regular_idx], min_ser_regular,
                color='blue', label=f"Min Regular SER: {min_ser_regular:.5f}")
    plt.scatter(erasure_threshold[min_ser_irregular_idx], min_ser_irregular,
                color='red', label=f"Min Irregular SER: {min_ser_irregular:.5f}")
    plt.xlabel("Erasure Threshold")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.yscale('log')
    plt.title("Symbol Error Rate vs. Erasure Threshold")
    plt.legend()
    plt.grid()

    # Plot Bit Error Rate (BER)
    plt.subplot(1, 2, 2)
    plt.plot(erasure_threshold, ber_regular,
             label="Regular LDPC", marker='o', color='blue')
    plt.plot(erasure_threshold, ber_irregular,
             label="Irregular LDPC", marker='s', color='red')
    plt.scatter(erasure_threshold[min_ber_regular_idx], min_ber_regular,
                color='blue', label=f"Min Regular BER: {min_ber_regular:.5f}")
    plt.scatter(erasure_threshold[min_ber_irregular_idx], min_ber_irregular,
                color='red', label=f"Min Irregular BER: {min_ber_irregular:.5f}")
    plt.xlabel("Erasure Threshold")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("Bit Error Rate vs. Erasure Threshold")
    plt.yscale('log')
    plt.legend()
    plt.grid()

    # Define the directory to save the plot
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the final plot
    filename = os.path.join(plot_dir, "final.png")
    plt.savefig(filename)
    plt.show()
