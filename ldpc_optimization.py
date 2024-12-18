import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Simulation function to evaluate BER
def simulate_ldpc_performance(H, G, snr_range, num_iterations=100):
    """
    Simulate LDPC encoding, transmission with noise, and decoding.
    Evaluates the average BER over a range of SNR values.

    Args:
    - H: Parity-check matrix
    - G: Generator matrix
    - snr_range: List of SNR values in dB
    - num_iterations: Number of iterations for each SNR value

    Returns:
    - Average BER across the SNR range
    """
    def single_iteration(snr_db):
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

    # Parallel computation for each SNR value
    ber_results = Parallel(n_jobs=-1)(delayed(single_iteration)(snr_db) for snr_db in snr_range)
    return np.mean(ber_results)

# Objective function for the optimizer
def fitness_function(params):
    """
    Objective function for genetic algorithm.
    params: [n, k, d_v, d_c]
    """
    n, k, d_v, d_c = int(params[0]), int(params[1]), int(params[2]), int(params[3])

    # Penalize invalid configurations
    if d_c <= d_v or n % d_c != 0 or k >= n:
        return 1.0  # High BER for invalid parameters

    try:
        H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
        snr_range = [5, 7, 10]  # Test across multiple SNR values
        ber = simulate_ldpc_performance(H, G, snr_range)
        return ber + 0.01 * (k / n)  # Penalize high code rate
    except:
        return 1.0  # Penalize failed configurations

# LDPC parameters
snr_db = 7  # Signal-to-Noise Ratio in dB

# Define bounds for each parameter
bounds = [
    (100, 1000),  # Codeword length (n)
    (50, 500),    # Information bits (k)
    (2, 6),       # Bit node degree (d_v)
    (6, 12)       # Check node degree (d_c)
]

# Run the genetic algorithm
ber_history = []

def logging_fitness_function(params):
    """
    Wraps fitness_function to log BER for each iteration.
    """
    ber = fitness_function(params)
    ber_history.append(ber)
    return ber

result = differential_evolution(
    func=logging_fitness_function,
    bounds=bounds,
    strategy="best1bin",  # Differential evolution strategy
    maxiter=50,           # Maximum number of generations
    popsize=20,           # Population size
    tol=1e-6,             # Tolerance for convergence
    mutation=(0.5, 1),    # Mutation factor range
    recombination=0.7,    # Recombination rate
    seed=42,              # Random seed for reproducibility
    disp=True             # Enables verbose output
)

# Extract the best solution
best_solution = result.x
best_ber = result.fun

# Print optimized parameters
print("\nOptimized Parameters:")
print(f"Codeword Length (n): {int(best_solution[0])}")
print(f"Information Bits (k): {int(best_solution[1])}")
print(f"Bit Node Degree (d_v): {int(best_solution[2])}")
print(f"Check Node Degree (d_c): {int(best_solution[3])}")
print(f"Best BER: {best_ber:.5e}")

# Plot BER convergence
plt.plot(ber_history)
plt.title("BER Convergence Over Generations")
plt.xlabel("Iteration")
plt.ylabel("BER")
plt.grid()
plt.show()
