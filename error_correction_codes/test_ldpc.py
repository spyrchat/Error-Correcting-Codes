import numpy as np


def validate_ldpc(H, G):
    """Validate the LDPC matrices H and G."""
    # Orthogonality check: H * G^T = 0 (mod 2)
    orthogonality_check = np.mod(H @ G.T, 2)
    if not np.all(orthogonality_check == 0):
        print("Validation failed: H * G^T != 0 (mod 2)")
        return False

    print("Validation successful: H and G are orthogonal.")
    return True


# Load the saved numpy array
try:
    H = np.load("H_matrix.npy")
    G = np.load("G_matrix.npy")
except FileNotFoundError:
    print("Error: One or both of the numpy files 'H_matrix.npy' and 'G_matrix.npy' were not found.")
    exit(1)
except Exception as e:
    print(f"Error loading numpy files: {e}")
    exit(1)

print("Loaded Parity-Check Matrix (H):")
print(H)

print("Loaded Generator Matrix (G):")
print(G)

if __name__ == "__main__":
    validate_ldpc(H, np.transpose(G))
