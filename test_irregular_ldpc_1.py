import numpy as np
# Import the function
import math
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from construct_irregular_ldpc import construct_irregular_ldpc


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


def gausselimination(A, b):
    """Solve linear system in Z/2Z via Gauss Gauss elimination."""
    if type(A) == scipy.sparse.csr_matrix:
        A = A.toarray().copy()
    else:
        A = A.copy()
    b = b.copy()
    n, k = A.shape

    for j in range(min(k, n)):
        listedepivots = [i for i in range(j, n) if A[i, j]]
        if len(listedepivots):
            pivot = np.min(listedepivots)
        else:
            continue
        if pivot != j:
            aux = (A[j, :]).copy()
            A[j, :] = A[pivot, :]
            A[pivot, :] = aux

            aux = b[j].copy()
            b[j] = b[pivot]
            b[pivot] = aux

        for i in range(j+1, n):
            if A[i, j]:
                A[i, :] = abs(A[i, :]-A[j, :])
                b[i] = abs(b[i]-b[j])

    return A, b


def gaussian_elimination_mod2(matrix):
    """
    Perform Gaussian elimination over GF(2) to reduce the matrix to row echelon form.

    Parameters:
        matrix (numpy.ndarray): A binary matrix (elements are 0 or 1).

    Returns:
        numpy.ndarray: Row echelon form of the input matrix over GF(2).
    """
    mat = matrix.copy()
    rows, cols = mat.shape
    pivot_row = 0

    for col in range(cols):
        # Debug: Print the matrix at each step
        print(f"Column {col}, Pivot Row {pivot_row}:")
        print(mat)

        # Find the row with a 1 in the current column at or below the pivot row
        for r in range(pivot_row, rows):
            if mat[r, col] == 1:
                # Swap the pivot row with the current row
                mat[[pivot_row, r]] = mat[[r, pivot_row]]
                print(f"Swapped rows {pivot_row} and {r}:")
                print(mat)
                break
        else:
            # No pivot in this column, move to the next column
            continue

        # Eliminate all 1s in the current column below the pivot row
        for r in range(pivot_row + 1, rows):
            if mat[r, col] == 1:
                mat[r] ^= mat[pivot_row]
                print(f"Row {r} XORed with Pivot Row {pivot_row}:")
                print(mat)

        # Move to the next pivot row
        pivot_row += 1

    return mat


def binaryproduct(X, Y):
    """Compute a matrix-matrix / vector product in Z/2Z."""
    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2


def get_message(tG, x):
    """
    Compute the original `n_bits` message from a `n_code` codeword `x`.

    Parameters:
    ----------
    tG: array (n_code, n_bits) coding matrix tG.
    x: array (n_code,) decoded codeword of length `n_code`.

    Returns:
    -------
    message: array (n_bits,). Original binary message.
    """
    n, k = tG.shape

    # Ensure x has the same size as the number of rows in tG
    if len(x) != n:
        raise ValueError(f"Inconsistent dimensions: x has {
                         len(x)} elements, but tG has {n} rows.")

    # Perform Gaussian elimination
    rtG, rx = gausselimination(tG, x)

    # Debugging: Check dimensions after Gaussian elimination
    print(f"rtG after Gaussian elimination: {rtG.shape}")
    print(f"rx shape after Gaussian elimination: {rx.shape}")

    # Truncate rtG and rx to ensure proper alignment with k
    rtG = rtG[:k, :k]
    rx = rx[:k]

    # Ensure rtG is in upper triangular form for back-substitution
    rtG = gaussian_elimination_mod2(rtG)

    # Perform back-substitution to compute the message
    message = np.zeros(k, dtype=int)
    for i in reversed(range(k)):
        message[i] = rx[i]
        if i + 1 < k:
            # XOR contributions from later message bits
            contribution = (np.dot(rtG[i, i + 1:], message[i + 1:]) % 2)
            print(f"Before back-substitution for message[{i}]: {message[i]}")
            message[i] ^= contribution
            print(f"After back-substitution for message[{i}]: {message[i]}")

    # Final debug for the decoded message
    print(f"Final decoded message: {message}")
    return message


def test_get_message():
    def binaryproduct(a, b):
        """Binary product for testing"""
        return np.dot(a, b) % 2

    # Test Cases
    tests = [
        {
            "name": "Basic Test",
            "tG": np.array([[1, 0], [1, 1]]),
            "x": np.array([1, 0]),
            "expected": np.array([1, 1]),
        },
        {
            "name": "Edge Case: Single-bit message",
            "tG": np.array([[1]]),
            "x": np.array([1]),
            "expected": np.array([1]),
        },
        {
            "name": "Valid Codeword",
            "tG": np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]),
            "x": np.array([1, 1, 0]),
            "expected": np.array([1, 1]),
        },
        {
            "name": "Mismatched Dimensions",
            "tG": np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]),
            "x": np.array([1, 0]),
            "expected": ValueError,
        },
    ]

    # Run Tests
    for test in tests:
        print()
        print(f"Running test: {test['name']}")
        try:
            if isinstance(test["expected"], type) and issubclass(test["expected"], Exception):
                try:
                    get_message(test["tG"], test["x"])
                except Exception as e:
                    assert isinstance(e, test["expected"]), f"Expected {
                        test['expected']}, but got {e}"
                    print(f"Test '{test['name']}' passed.")
            else:
                result = get_message(test["tG"], test["x"])
                assert np.array_equal(result[:len(test["expected"])], test["expected"]), f"Expected {
                    test['expected']}, but got {result}"
                print(f"Test '{test['name']}' passed.")
        except Exception as e:
            print(f"Test '{test['name']}' failed with error: {e}")


def test_gaussian_elimination_and_message_retrieval():
    test_cases = [
        {
            "name": "Basic Test",
            "tG": np.array([[1, 0], [0, 1]], dtype=int),
            "x": np.array([1, 0], dtype=int),
            "expected_message": np.array([1, 0], dtype=int),
        },
        {
            "name": "Edge Case: Single-bit message",
            "tG": np.array([[1]], dtype=int),
            "x": np.array([1], dtype=int),
            "expected_message": np.array([1], dtype=int),
        },
        {
            "name": "Valid Codeword",
            "tG": np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int),
            "x": np.array([1, 1, 0], dtype=int),
            # Only the first two bits
            "expected_message": np.array([1, 1], dtype=int),
        },
        {
            "name": "Mismatched Dimensions",
            "tG": np.array([[1, 0], [0, 1]], dtype=int),
            "x": np.array([1, 0, 1], dtype=int),
            "expected_error": ValueError,
        },
    ]

    for test in test_cases:
        print()
        print(f"Running test: {test['name']}")
        try:
            result = get_message(test["tG"], test["x"])
            assert np.array_equal(result[:len(test["expected_message"])], test["expected_message"]), (
                f"Test '{test['name']}' failed with error: "
                f"Expected {test['expected_message']}, but got {result}"
            )
            print(f"Test '{test['name']}' passed.")
        except Exception as e:
            if "expected_error" in test and isinstance(e, test["expected_error"]):
                print(f"Test '{test['name']}' passed.")
            else:
                print(f"Test '{test['name']}' failed with error: {e}")


def test_gaussian_elimination_mod2():
    tests = [
        {
            "name": "Basic Test",
            "input": np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
            "expected": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]]),
        },
        {
            "name": "All Zero Rows",
            "input": np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
            "expected": np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]]),
        },
        {
            "name": "Fully Dense Matrix",
            "input": np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]]),
            "expected": np.array([[1, 1, 1], [0, 1, 0], [0, 0, 1]]),
        },
    ]

    for test in tests:
        print()
        print(f"Running test: {test['name']}")
        result = gaussian_elimination_mod2(test["input"])
        if np.array_equal(result, test["expected"]):
            print(f"Test '{test['name']}' passed.")
        else:
            print(f"Test '{test['name']}' failed.")
            print("Input:")
            print(test["input"])
            print("Result:")
            print(result)
            print("Expected:")
            print(test["expected"])


if __name__ == "__main__":
    test_get_message()
    print('test_get_message finished')
    test_gaussian_elimination_and_message_retrieval()
    print('test_gaussian_elimination_and_message_retrieval finished')
    test_gaussian_elimination_mod2()
    print('test_gaussian_elimination_mod2 finished')
