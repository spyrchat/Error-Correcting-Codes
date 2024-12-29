import numpy as np
from error_correction_codes.construct_irregular_ldpc import construct_irregular_ldpc  # Import the function

import math
import numpy as np
from scipy.sparse import csr_matrix
from error_correction_codes.construct_irregular_ldpc import construct_irregular_ldpc

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

def validate_ldpc(H, G):
    """
    Validate the LDPC matrices H and G by checking:
    1. Orthogonality: H * G^T = 0 (mod 2).
    2. Codeword validity using G.

    Parameters:
    H (scipy.sparse.csr_matrix): Parity-check matrix.
    G (scipy.sparse.csr_matrix): Generator matrix.

    Returns:
    bool: True if the matrices are valid, False otherwise.
    """
    H_dense = H.toarray()
    G_dense = G.toarray()

    # Orthogonality check
    orthogonality_check = np.mod(H_dense @ G_dense.T, 2)
    if not np.all(orthogonality_check == 0):
        print("Validation failed: H * G^T != 0 (mod 2)")
        return False

    print("Validation successful: H and G are orthogonal.")

    # Codeword validity
    k = G_dense.shape[0]
    message = np.random.randint(0, 2, size=k)
    codeword = np.mod(message @ G_dense, 2)
    parity_check_result = np.mod(H_dense @ codeword.T, 2)

    if not np.all(parity_check_result == 0):
        print("Validation failed: Generated codeword does not satisfy H.")
        return False

    print("Validation successful: Generated codeword satisfies H.")
    return True

def main():
    # Parameters from LDPC design
    n = 20000  # Increased to ensure m < n
    c_avg = 15.475  # Average check node degree

    # Adjusted Lambda and rho distributions
    Lambda = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.01, 0.02, 0.03, 0.04, 0.05, 0.8
    ])
    Lambda /= Lambda.sum()

    rho = np.array([0.4, 0.6])
    rho /= rho.sum()

    # Debugging outputs
    Lambda_prime = np.dot(np.arange(1, len(Lambda) + 1), Lambda)
    rho_prime = np.dot(np.arange(1, len(rho) + 1), rho)
    print(f"\nLambda'(1): {Lambda_prime}, Rho'(1): {rho_prime}")
    print(f"Expected Design Rate (R): {1 - (rho_prime / Lambda_prime)}")

    # Generate the LDPC matrices
    print("\nGenerating the LDPC matrices...")
    try:
        H, G = construct_irregular_ldpc(n, Lambda, rho)

        print("\nParity-check matrix H:")
        print(H.toarray())

        print("\nGenerator matrix G:")
        print(G.toarray())

        # Validate the matrices
        print("\nValidating LDPC matrices...")
        if validate_ldpc(H, G):
            print("\nLDPC design is valid.")
        else:
            print("\nLDPC design is invalid. Check the design parameters.")

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


