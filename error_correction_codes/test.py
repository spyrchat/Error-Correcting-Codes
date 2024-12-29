from utils import _bitsandnodes
import numpy as np  
def test_bitsandnodes():
    """Test the bitsandnodes function with a sample H matrix."""
    # Example parity-check matrix
    H = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0]
    ])

    bits_hist, bits_values, nodes_hist, nodes_values = _bitsandnodes(H)

    print("Bits histogram (row sums):", bits_hist)
    print("Bits values (row connections):", bits_values)
    print("Nodes histogram (column sums):", nodes_hist)
    print("Nodes values (column connections):", nodes_values)

# Run the test
test_bitsandnodes()

from decoder import get_message

def test_get_message():
    # Example parameters
    tG = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])  # Corrected: k=3
    x = np.array([1, 0, 1])  # Decoded codeword with length matching tG's columns

    # Get original message
    message = get_message(tG, x)

    # Assertions
    assert message.shape == (3,), f"Message shape mismatch: {message.shape}"
    print(f"get_message passed. message={message}")

test_get_message()

from erasure_channel_encoding_irregular import simulate_irregular_ldpc_erasure_correction

def test_simulate_irregular_ldpc_erasure_correction():
    # Example parameters
    erasure_thresholds = [0.1, 0.5, 0.9]
    n = 3213
    Lambda = [0.2, 0.3, 0.5]
    rho = [0.6, 0.4]
    snr_db = 5  # Signal-to-noise ratio

    # Run simulation
    ser_results, bit_rate_results = simulate_irregular_ldpc_erasure_correction(erasure_thresholds, n, Lambda, rho, snr_db=snr_db)

    # Assertions
    assert len(ser_results) == len(erasure_thresholds), f"SER results length mismatch"
    assert len(bit_rate_results) == len(erasure_thresholds), f"Bit rate results length mismatch"
    print("simulate_irregular_ldpc_erasure_correction passed.")

test_simulate_irregular_ldpc_erasure_correction()
