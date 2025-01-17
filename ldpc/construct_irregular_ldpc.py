import numpy as np
from scipy.sparse import csr_matrix
from collections import deque
from utils import gaussjordan


def find_smallest(array):
    """Find the index of the smallest element in the array."""
    return np.argmin(array)


class PEG:
    """
    Progressive edge growth algorithm for generating LDPC matrices.
    """

    def __init__(self, nvar, nchk, degree_sequence):
        self.degree_sequence = degree_sequence
        self.nvar = nvar
        self.nchk = nchk
        self.H = np.zeros((nchk, nvar), dtype=np.int32)
        self.sym_degrees = np.zeros(nvar, dtype=np.int32)
        self.chk_degrees = np.zeros(nchk, dtype=np.int32)

    def grow_edge(self, var, chk):
        """Grow an edge between a variable and check node."""
        self.H[chk, var] = 1
        self.sym_degrees[var] += 1
        self.chk_degrees[chk] += 1

    def bfs(self, var):
        """Perform BFS to find the smallest degree check node."""
        var_list = np.zeros(self.nvar, dtype=bool)
        var_list[var] = True

        cur_chk_list = np.zeros(self.nchk, dtype=bool)
        new_chk_list = np.zeros(self.nchk, dtype=bool)

        var_queue = deque([var])

        while True:
            # Process variable nodes and find connected check nodes
            for v in var_queue:
                connected_checks = np.where(self.H[:, v] == 1)[0]
                new_chk_list[connected_checks] = True

            var_queue.clear()

            # Process check nodes and find connected variable nodes
            for chk in np.where(new_chk_list & ~cur_chk_list)[0]:
                connected_vars = np.where(self.H[chk, :] == 1)[0]
                var_queue.extend(v for v in connected_vars if not var_list[v])
                var_list[connected_vars] = True

            cur_chk_list = new_chk_list.copy()

            if np.all(new_chk_list):
                # All check nodes visited, return the smallest degree check node
                return self.find_smallest_chk(cur_chk_list)
            elif not np.any(new_chk_list ^ cur_chk_list):
                # No new check nodes found, return the smallest degree check node
                return self.find_smallest_chk(cur_chk_list)

    def find_smallest_chk(self, cur_chk_list):
        """Find the smallest degree check node among unvisited nodes."""
        unvisited_checks = np.where(~cur_chk_list)[0]
        if len(unvisited_checks) == 0:
            raise ValueError("No unvisited check nodes available.")
        return unvisited_checks[np.argmin(self.chk_degrees[unvisited_checks])]

    def progressive_edge_growth(self):
        """Perform progressive edge growth to generate the LDPC matrix."""
        for var in range(self.nvar):
            for k in range(self.degree_sequence[var]):
                if k == 0:
                    smallest_degree_chk = find_smallest(self.chk_degrees)
                    self.grow_edge(var, smallest_degree_chk)
                else:
                    chk = self.bfs(var)
                    self.grow_edge(var, chk)


def coding_matrix(H, sparse=True):
    """Generate the coding matrix G given the LDPC matrix H."""
    if isinstance(H, csr_matrix):
        H = H.toarray()
    n_equations, n_code = H.shape

    # Gaussian elimination
    Href_colonnes, tQ = gaussjordan(H.T, 1)
    Href_diag = gaussjordan(Href_colonnes.T)
    Q = tQ.T
    n_bits = n_code - Href_diag.sum()

    # Create the identity matrix
    Y = np.zeros((n_code, n_bits), dtype=int)
    Y[n_code - n_bits:, :] = np.identity(n_bits, dtype=int)

    if sparse:
        Q = csr_matrix(Q)
        Y = csr_matrix(Y)

    # Convert to dense arrays for modulo operation
    Q_dense = Q.toarray() if sparse else Q
    Y_dense = Y.toarray() if sparse else Y

    # Perform the modulo operation
    tG_dense = np.mod(Q_dense @ Y_dense, 2)

    # Convert back to sparse if needed
    if sparse:
        return csr_matrix(tG_dense)
    return tG_dense


def validate_ldpc(H, G):
    """Validate the LDPC matrices H and G."""
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

    peg_instance = PEG(nvar=N, nchk=M, degree_sequence=degree_sequence)
    peg_instance.progressive_edge_growth()

    print("Generated Parity-Check Matrix (H):")
    print(peg_instance.H)

    np.save("H_matrix.npy", peg_instance.H)
    print("Parity-Check Matrix saved as 'H_matrix.npy'")

    G = coding_matrix(peg_instance.H)

    np.save("G_matrix.npy", G)
    print("Generator Matrix saved as 'G_matrix.npy'")

    if validate_ldpc(peg_instance.H, np.transpose(G)):
        print("LDPC matrices are valid.")
    else:
        print("LDPC matrices are invalid.")
