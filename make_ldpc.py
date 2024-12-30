import numpy as np
from scipy.sparse import csr_matrix
import utils


def parity_check_matrix_irregular(n_code, lambda_dist, rho_dist, seed=None):
    """
    Build an irregular Parity-Check Matrix H based on degree distributions.

    Parameters
    ----------
    n_code: int, Length of the codewords.
    lambda_dist: list, Variable node degree distribution.
    rho_dist: list, Check node degree distribution.
    seed: int, seed of the random generator.

    Returns
    -------
    H: array (n_equations, n_code). Irregular LDPC matrix H.
    """
    rng = np.random.default_rng(seed)

    # Calculate total number of edges
    n_edges = int(n_code * sum([(i + 1) * lambda_dist[i]
                  for i in range(len(lambda_dist))]))
    n_equations = int(
        n_edges / sum([(j + 1) * rho_dist[j] for j in range(len(rho_dist))]))

    H = np.zeros((n_equations, n_code), dtype=int)

    # Assign edges to variable nodes
    variable_nodes = []
    for i, frac in enumerate(lambda_dist):
        variable_nodes.extend([i + 1] * int(frac * n_edges))
    rng.shuffle(variable_nodes)

    # Assign edges to check nodes
    check_nodes = []
    for j, frac in enumerate(rho_dist):
        check_nodes.extend([j + 1] * int(frac * n_edges))
    rng.shuffle(check_nodes)

    # Connect variable nodes to check nodes
    edge_index = 0
    for edge in range(n_edges):
        v_node = edge_index % n_code
        c_node = edge_index % n_equations
        H[c_node, v_node] = 1
        edge_index += 1

    return H


def coding_matrix(H, sparse=True):
    """Return the generating coding matrix G given the LDPC matrix H.

    Parameters
    ----------
    H: array (n_equations, n_code). Parity check matrix of an LDPC code.
    sparse: bool, default True. Use sparse matrices for large codes.

    Returns
    -------
    G.T: array (n_bits, n_code). Transposed coding matrix.
    """
    if type(H) == csr_matrix:
        H = H.toarray()
    n_equations, n_code = H.shape

    Href_colonnes, tQ = utils.gaussjordan(H.T, 1)
    Href_diag = utils.gaussjordan(np.transpose(Href_colonnes))
    Q = tQ.T
    n_bits = n_code - Href_diag.sum()

    Y = np.zeros(shape=(n_code, n_bits)).astype(int)
    Y[n_code - n_bits:, :] = np.identity(n_bits)

    if sparse:
        Q = csr_matrix(Q)
        Y = csr_matrix(Y)

    tG = utils.binaryproduct(Q, Y)

    return tG


def coding_matrix_systematic(H, sparse=True):
    """Compute a coding matrix G in systematic format with an identity block.

    Parameters
    ----------
    H: array (n_equations, n_code). Parity-check matrix.
    sparse: bool, default True. Use sparse matrices for large codes.

    Returns
    -------
    H_new: array (n_equations, n_code). Modified parity-check matrix.
    G_systematic.T: Transposed Systematic Coding matrix.
    """
    n_equations, n_code = H.shape

    P1 = np.identity(n_code, dtype=int)
    Hrowreduced = utils.gaussjordan(H)
    n_bits = n_code - sum([a.any() for a in Hrowreduced])

    while True:
        zeros = [i for i in range(min(n_equations, n_code))
                 if not Hrowreduced[i, i]]
        if len(zeros):
            indice_colonne_a = min(zeros)
        else:
            break
        list_ones = [j for j in range(
            indice_colonne_a + 1, n_code) if Hrowreduced[indice_colonne_a, j]]
        if len(list_ones):
            indice_colonne_b = min(list_ones)
        else:
            break
        aux = Hrowreduced[:, indice_colonne_a].copy()
        Hrowreduced[:, indice_colonne_a] = Hrowreduced[:, indice_colonne_b]
        Hrowreduced[:, indice_colonne_b] = aux

        aux = P1[:, indice_colonne_a].copy()
        P1[:, indice_colonne_a] = P1[:, indice_colonne_b]
        P1[:, indice_colonne_b] = aux

    P1 = P1.T
    identity = list(range(n_code))
    sigma = identity[n_code - n_bits:] + identity[:n_code - n_bits]

    P2 = np.zeros(shape=(n_code, n_code), dtype=int)
    P2[identity, sigma] = np.ones(n_code)

    if sparse:
        P1 = csr_matrix(P1)
        P2 = csr_matrix(P2)
        H = csr_matrix(H)

    P = utils.binaryproduct(P2, P1)

    if sparse:
        P = csr_matrix(P)

    H_new = utils.binaryproduct(H, np.transpose(P))

    G_systematic = np.zeros((n_bits, n_code), dtype=int)
    G_systematic[:, :n_bits] = np.identity(n_bits)
    G_systematic[:, n_bits:] = (
        Hrowreduced[:n_code - n_bits, n_code - n_bits:]).T

    return H_new, G_systematic.T


def make_ldpc_irregular(n_code, lambda_dist=None, rho_dist=None, systematic=False, sparse=True, seed=None):
    """
    Create an irregular LDPC coding and decoding matrices H and G.

    Parameters
    ----------
    n_code: int, Length of the codewords.
    lambda_dist: list, Variable node degree distribution.
    rho_dist: list, Check node degree distribution.
    systematic: bool, default False. If True, constructs a systematic coding matrix G.
    sparse: bool, default True. Use sparse matrices for large codes.
    seed: int, Random seed for reproducibility.

    Returns
    -------
    H: scipy.sparse.csr_matrix, Parity-check matrix.
    G: scipy.sparse.csr_matrix, Generator matrix.
    """
    H = parity_check_matrix_irregular(n_code, lambda_dist, rho_dist, seed)

    if systematic:
        H, G = coding_matrix_systematic(H, sparse=sparse)
    else:
        G = coding_matrix(H, sparse=sparse)

    return H, G
