from PIL import Image
import matplotlib.pyplot as plt

# Load the two images (update with your file paths)
regular_img = Image.open(r"ldpc\plots\results_snr_10 n=3205.png")
# Replace with your irregular LDPC image path
irregular_img = Image.open(r"ldpc\plots\image.png")

# Create a new figure
plt.figure(figsize=(12, 6))

# Add the first image (regular LDPC)
plt.subplot(1, 2, 1)
plt.imshow(regular_img)
plt.axis('off')  # Hide axes
plt.title("Regular LDPC")

# Add the second image (irregular LDPC)
plt.subplot(1, 2, 2)
plt.imshow(irregular_img)
plt.axis('off')  # Hide axes
plt.title("Irregular LDPC")

# Show the combined figure
plt.tight_layout()
plt.show()
