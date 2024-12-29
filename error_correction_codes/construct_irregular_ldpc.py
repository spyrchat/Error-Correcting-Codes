import numpy as np
from scipy.sparse import csr_matrix


def gaussian_elimination_mod2(H):
    """Reduce a binary matrix H to systematic form [I | P]."""
    H = H.copy()
    rows, cols = H.shape
    pivot_row = 0

    for col in range(cols):
        for r in range(pivot_row, rows):
            if H[r, col] == 1:
                H[[pivot_row, r]] = H[[r, pivot_row]]
                break
        else:
            continue

        for r in range(rows):
            if r != pivot_row and H[r, col] == 1:
                H[r] ^= H[pivot_row]

        pivot_row += 1
        if pivot_row == rows:
            break

    return H


def construct_irregular_ldpc(n, Lambda, rho):
    """
    Constructs an irregular LDPC parity-check matrix and generator matrix.

    Parameters:
        n (int): Number of variable nodes.
        Lambda (list): Coefficients of the variable node degree distribution polynomial.
        rho (list): Coefficients of the check node degree distribution polynomial.

    Returns:
        H (scipy.sparse.csr_matrix): The parity-check matrix.
        G (scipy.sparse.csr_matrix): The generator matrix.
    """
    Lambda_prime = np.dot(np.arange(1, len(Lambda) + 1), Lambda[::-1])  # Reversed for correct order
    rho_prime = np.dot(np.arange(1, len(rho) + 1), rho[::-1])  # Reversed for correct order
    m = int(np.floor((Lambda_prime / rho_prime) * n))

    print(f"Computed Check Nodes (m): {m}, Variable Nodes (n): {n}")
    if m >= n:
        raise ValueError(f"Invalid parameters: m = {m} exceeds or equals n = {n}.")

    variable_sockets = []
    check_sockets = []

    for i in range(n):
        degree = np.random.choice(np.arange(1, len(Lambda) + 1), p=Lambda[::-1])
        variable_sockets.extend([i] * degree)

    for j in range(m):
        degree = np.random.choice(np.arange(1, len(rho) + 1), p=rho[::-1])
        check_sockets.extend([j] * degree)

    np.random.shuffle(variable_sockets)
    np.random.shuffle(check_sockets)
    min_length = min(len(variable_sockets), len(check_sockets))
    variable_sockets = variable_sockets[:min_length]
    check_sockets = check_sockets[:min_length]

    H = np.zeros((m, n), dtype=int)
    for v, c in zip(variable_sockets, check_sockets):
        H[c, v] = 1

    row_degrees = H.sum(axis=1)
    col_degrees = H.sum(axis=0)
    for i in range(m):
        if row_degrees[i] == 0:
            v = np.random.randint(0, n)
            H[i, v] = 1
    for j in range(n):
        if col_degrees[j] == 0:
            c = np.random.randint(0, m)
            H[c, j] = 1

    H_sparse = csr_matrix(H)

    H_dense = H_sparse.toarray()
    try:
        H_systematic = gaussian_elimination_mod2(H_dense)
        m, n = H_systematic.shape
        k = n - m
        if k <= 0:
            raise ValueError("Number of message bits is invalid. Check input parameters.")
        P = H_systematic[:, m:]
        I_k = np.eye(k, dtype=int)
        G_dense = np.hstack((P.T, I_k))
        G_sparse = csr_matrix(G_dense)
    except Exception as e:
        raise ValueError(f"Systematic form transformation failed: {e}")

    return H_sparse, G_sparse


def main():
    # Corrected Lambda(x) coefficients (highest order first)
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

    print(f"Lambda'(1): {Lambda_prime}")
    print(f"Rho'(1): {rho_prime}")
    print(f"Design Rate (R): {design_rate}")
    print(f"Computed n: {n}")

    try:
        H, G = construct_irregular_ldpc(n, Lambda, rho)
        print("\nParity-check matrix H:")
        print(H.toarray())
        print("\nGenerator matrix G:")
        print(G.toarray())
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
