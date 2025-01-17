import unittest
import numpy as np
from scipy.sparse import csr_matrix
from construct_irregular_ldpc import PEG, coding_matrix, validate_ldpc


class TestPEG(unittest.TestCase):

    def setUp(self):
        """
        Set up common parameters for the tests.
        """
        self.Lambda = [2.066e-06, 1.96e-06, 1.816e-06, 1.612e-06,
                       1.317e-06, 0.2824, 0.0997, 0.1735, 0.4444, 0]
        self.Rho = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.design_rate = 0.723

        Lambda_prime = np.dot(
            np.arange(1, len(self.Lambda) + 1), np.flip(self.Lambda))
        Rho_prime = np.dot(np.arange(1, len(self.Rho) + 1), np.flip(self.Rho))

        self.N = int(np.ceil((Rho_prime / (1 - self.design_rate)) ** 2))
        self.M = int(np.floor((Lambda_prime / Rho_prime) * self.N))

        self.degree_sequence = np.random.choice(
            np.arange(1, len(self.Lambda) + 1), size=self.N, p=self.Lambda / np.sum(self.Lambda))

    def test_parity_check_matrix_generation(self):
        """
        Test if the PEG algorithm generates a valid parity-check matrix H.
        """
        peg_instance = PEG(nvar=self.N, nchk=self.M,
                           degree_sequence=self.degree_sequence)
        peg_instance.progressive_edge_growth()
        H = peg_instance.H

        # Validate dimensions
        self.assertEqual(H.shape, (self.M, self.N))

        # Validate degree sequence
        sym_degrees = H.sum(axis=0)
        self.assertTrue(np.all(sym_degrees == self.degree_sequence))

        # Validate non-zero elements in check node rows
        chk_degrees = H.sum(axis=1)
        self.assertTrue(np.all(chk_degrees > 0))

    def test_generator_matrix(self):
        """
        Test if the generator matrix G is orthogonal to H.
        """
        peg_instance = PEG(nvar=self.N, nchk=self.M,
                           degree_sequence=self.degree_sequence)
        peg_instance.progressive_edge_growth()
        H = peg_instance.H

        # Generate the generator matrix G
        G = coding_matrix(H)

        # Validate dimensions
        self.assertEqual(G.shape[0], H.shape[1])
        self.assertLess(G.shape[1], G.shape[0])

        # Validate orthogonality
        self.assertTrue(validate_ldpc(H, np.transpose(G)))

    def test_sparse_matrix_support(self):
        """
        Test if sparse matrix inputs work for coding_matrix.
        """
        peg_instance = PEG(nvar=self.N, nchk=self.M,
                           degree_sequence=self.degree_sequence)
        peg_instance.progressive_edge_growth()
        H_sparse = csr_matrix(peg_instance.H)

        # Generate the generator matrix G
        G_sparse = coding_matrix(H_sparse)

        # Ensure G is a sparse matrix
        self.assertIsInstance(G_sparse, csr_matrix)

        # Validate orthogonality
        self.assertTrue(validate_ldpc(H_sparse.toarray(),
                        np.transpose(G_sparse.toarray())))


if __name__ == "__main__":
    unittest.main()
