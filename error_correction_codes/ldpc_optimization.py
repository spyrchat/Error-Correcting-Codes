import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plot
import math

# Returns rho polynomial (highest exponents first) corresponding to average check node degree c_avg


def c_avg_to_rho(c_avg):
    """
    Converts the average check node degree c_avg into a check node polynomial rho.

    Parameters:
    c_avg (float): Average check node degree.

    Returns:
    numpy.ndarray: Polynomial coefficients of rho(x).
    """
    ct = math.floor(c_avg)
    r1 = ct * (ct + 1 - c_avg) / c_avg
    r2 = (c_avg - ct * (ct + 1 - c_avg)) / c_avg
    rho_poly = np.concatenate(([r2, r1], np.zeros(ct - 1)))
    return rho_poly

# Finds the optimal variable node degree distribution lambda for a given epsilon, v_max, and c_avg


def find_best_lambda(epsilon, v_max, c_avg):
    """
    Optimizes the variable node degree distribution (lambda) for given parameters.

    Parameters:
    epsilon (float): Channel parameter.
    v_max (int): Maximum variable node degree.
    c_avg (float): Average check node degree.

    Returns:
    numpy.ndarray: Optimal lambda distribution.
    """
    rho = c_avg_to_rho(c_avg)
    # Quantization of fixed-point condition
    D = 500
    xi_range = np.arange(1.0, D + 1, 1) / D

    # Variable to optimize is lambda with v_max entries
    v_lambda = cp.Variable(shape=v_max)

    # Objective function
    cv = 1 / np.arange(v_max, 0, -1)
    objective = cp.Maximize(v_lambda @ cv)

    # Constraints
    # Constraint 1: v_lambda are fractions between 0 and 1 and sum up to 1
    constraints = [cp.sum(v_lambda) == 1, v_lambda >= 0]

    # Constraint 2: No variable nodes of degree 1
    constraints += [v_lambda[v_max - 1] == 0]

    # Constraint 3: Fixed-point condition for all discrete xi values
    for xi in xi_range:
        constraints += [v_lambda @ [epsilon * (1 - np.polyval(rho, 1.0 - xi)) ** (
            v_max - 1 - j) for j in range(v_max)] - xi <= 0]

    # Constraint 4: Stability condition
    constraints += [v_lambda[v_max - 2] <= 1 /
                    epsilon / np.polyval(np.polyder(rho), 1.0)]

    # Set up the problem and solve
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=True)
    except cp.error.SolverError:
        print("Solver ECOS failed. Trying SCS...")
        problem.solve(solver=cp.SCS, verbose=True)

    if problem.status == "optimal":
        r_lambda = v_lambda.value
        # Remove entries close to zero and renormalize
        r_lambda[r_lambda <= 1e-7] = 0
        r_lambda = r_lambda / sum(r_lambda)
    else:
        r_lambda = np.array([])

    return r_lambda

# Finds the best rate for a given epsilon, v_max, and maximum check node degree


def find_best_rate(epsilon, v_max, c_max):
    """
    Computes the best design rate for given parameters by optimizing lambda and rho.

    Parameters:
    epsilon (float): Channel parameter.
    v_max (int): Maximum variable node degree.
    c_max (int): Maximum check node degree.

    Returns:
    tuple: Best achievable design rate, best lambda, and corresponding c_avg.
    """
    c_range = np.linspace(3, c_max, num=100)
    rates = np.zeros_like(c_range)
    best_lambda = None
    best_c_avg = None

    # Loop over all c_avg values
    for index, c_avg in enumerate(c_range):
        p_lambda = find_best_lambda(epsilon, v_max, c_avg)
        p_rho = c_avg_to_rho(c_avg)
        if np.array(p_lambda).size > 0:
            design_rate = 1 - \
                np.polyval(np.polyint(p_rho), 1) / \
                np.polyval(np.polyint(p_lambda), 1)
            if design_rate >= 0:
                rates[index] = design_rate

    # Find largest rate
    largest_rate_index = np.argmax(rates)
    best_lambda = find_best_lambda(epsilon, v_max, c_range[largest_rate_index])
    best_c_avg = c_range[largest_rate_index]
    best_rate = rates[largest_rate_index]

    return best_rate, best_lambda, best_c_avg


if __name__ == "__main__":
    # Main optimization loop
    target_rate = 0.7
    dv_max = 16
    dc_max = 22

    T_Delta = 0.001
    epsilon = 0.5
    Delta_epsilon = 0.5

    best_solution = None
    best_threshold = None

    while Delta_epsilon >= T_Delta:
        print('Running optimization for epsilon = %1.5f' % epsilon)

        rate, lambda_poly, c_avg = find_best_rate(epsilon, dv_max, dc_max)
        if rate > target_rate:
            epsilon = epsilon + Delta_epsilon / 2
        else:
            epsilon = epsilon - Delta_epsilon / 2

        if best_solution is None or rate > best_threshold:
            best_solution = (rate, lambda_poly, c_avg, epsilon)
            best_threshold = rate

        Delta_epsilon = Delta_epsilon / 2

    # Print the best solution
    if best_solution:
        print("\nBest solution found:")
        print("Design Rate: %1.3f" % best_solution[0])
        print("Lambda Polynomial:")
        print(np.poly1d(best_solution[1], variable='Z'))
        print("Average Check Node Degree: %1.3f" % best_solution[2])
        print("Epsilon: %1.5f" % best_solution[3])
        print(c_avg_to_rho(best_solution[2]))
