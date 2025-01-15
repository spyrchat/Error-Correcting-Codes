import utils
from scipy.sparse import csr_matrix
from collections import deque
import copy
import numpy as np
from utils import gaussjordan


def find_smallest(array):
    if len(array) == 1:
        return 0
    elif len(array) == 2:
        if array[0] <= array[1]:
            return 0
        else:
            return 1
    else:
        mid = len(array) // 2
        arrayA = array[:mid]
        arrayB = array[mid:]
        smallA = find_smallest(arrayA)
        smallB = find_smallest(arrayB)
        if arrayA[smallA] <= arrayB[smallB]:
            return smallA
        else:
            return mid + smallB


class peg():

    """
    Progressive edge growth algorithm for generating
    LDPC matrices. The algorithm is obtained from [1]
    """

    def __init__(self, nvar, nchk, degree_sequence, verbose=True):
        self.degree_sequence = degree_sequence
        self.nvar = nvar
        self.nchk = nchk
        self.H = np.zeros((nchk, nvar), dtype=np.int32)
        self.sym_degrees = np.zeros(nvar, dtype=np.int32)
        self.chk_degrees = np.zeros(nchk, dtype=np.int32)
        self.verbose = verbose

    def grow_edge(self, var, chk):
        self.H[chk, var] = 1
        self.sym_degrees[var] += 1
        self.chk_degrees[chk] += 1

    def bfs(self, var):
        var_list = np.zeros(self.nvar, dtype=np.int32)
        var_list[var] = 1
        cur_chk_list = np.zeros(self.nchk, dtype=np.int32)
        queue = deque([var])

        while queue:
            current_var = queue.popleft()
            for chk in np.where(self.H[:, current_var] == 1)[0]:
                if cur_chk_list[chk] == 0:
                    cur_chk_list[chk] = 1
                    for next_var in np.where(self.H[chk, :] == 1)[0]:
                        if var_list[next_var] == 0:
                            var_list[next_var] = 1
                            queue.append(next_var)

        return self.find_smallest_chk(cur_chk_list)

    def find_smallest_chk(self, cur_chk_list):
        available_indices = np.where(cur_chk_list == 0)[0]
        if len(available_indices) == 0:
            if self.verbose:
                print("No available check nodes, forcing connection.")
            return np.argmin(self.chk_degrees)
        available_degrees = self.chk_degrees[available_indices]
        return available_indices[np.argmin(available_degrees)]

    def progressive_edge_growth(self):
        for var in range(self.nvar):
            if self.verbose:
                print(f"Growing edges for variable {var}")
            for k in range(self.degree_sequence[var]):
                if self.verbose:
                    print(f"Attempting connection {k + 1} for variable {var}")
                try:
                    if k == 0:
                        smallest_degree_chk = np.argmin(self.chk_degrees)
                        self.grow_edge(var, smallest_degree_chk)
                    else:
                        chk = self.bfs(var)
                        self.grow_edge(var, chk)
                except ValueError as e:
                    print(f"Error for variable {var}, edge {k + 1}: {e}")
                    raise


def coding_matrix(H, sparse=True):
    """Return the generating coding matrix G given the LDPC matrix H."""
    if type(H) == csr_matrix:
        H = H.toarray()
    n_equations, n_code = H.shape

    Href_colonnes, tQ = gaussjordan(H.T, 1)
    Href_diag = gaussjordan(np.transpose(Href_colonnes))
    Q = tQ.T
    n_bits = n_code - Href_diag.sum()

    Y = np.zeros(shape=(n_code, n_bits)).astype(int)
    Y[n_code - n_bits:, :] = np.identity(n_bits)

    if sparse:
        Q = csr_matrix(Q)
        Y = csr_matrix(Y)

    tG = utils.binaryproduct(Q, Y)
    return tG


# Validation


def validate_ldpc(H, G):
    """Validate the LDPC matrices H and G."""
    # Orthogonality check: H * G^T = 0 (mod 2)
    orthogonality_check = np.mod(H @ G.T, 2)
    if not np.all(orthogonality_check == 0):
        print("Validation failed: H * G^T != 0 (mod 2)")
        return False

    print("Validation successful: H and G are orthogonal.")
    return True


if __name__ == "__main__":
    # Example Usage
    Lambda = [2.066e-06, 1.96e-06, 1.816e-06, 1.612e-06,
              1.317e-06, 0.2824, 0.0997, 0.1735, 0.4444, 0]
    Rho = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    design_rate = 0.723

    Lambda_prime = np.dot(np.arange(1, len(Lambda) + 1), np.flip(Lambda))
    Rho_prime = np.dot(np.arange(1, len(Rho) + 1), np.flip(Rho))

    N = int(np.ceil((Rho_prime / (1 - design_rate)) ** 2))
    M = int(np.floor((Lambda_prime / Rho_prime) * N))

    degree_sequence = np.random.choice(
        np.arange(1, len(Lambda) + 1), size=N, p=Lambda / np.sum(Lambda))

    peg_instance = peg(nvar=N, nchk=M, degree_sequence=degree_sequence)
    peg_instance.progressive_edge_growth()

    print("Generated Parity-Check Matrix (H):")
    print(peg_instance.H)

    # Save H as a numpy array
    np.save("H_matrix.npy", peg_instance.H)
    print("Parity-Check Matrix saved as 'H_matrix.npy'")

    # Generate the generator matrix G
    G = coding_matrix(peg_instance.H)

    # Save G as a numpy array
    np.save("G_matrix.npy", G)
    print("Generator Matrix saved as 'G_matrix.npy'")

    if validate_ldpc(peg_instance.H, np.transpose(G)):
        print("LDPC matrices are valid.")
    else:
        print("LDPC matrices are invalid.")
