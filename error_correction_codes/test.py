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
