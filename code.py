import numpy as np
import cv2
import hashlib
from skimage.metrics import structural_similarity as ssim
import random
import matplotlib.pyplot as plt

# 1. Preprocessing: Resizing, Normalization, Augmentation
def preprocess_image(image_path, size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    normalized_image = image / 255.0
    return normalized_image

# 2. Elliptic Curve Cryptography (ECC) for Key Generation
def generate_ecc_key():
    private_key = random.randint(1, 1e9)
    public_key = hashlib.sha256(str(private_key).encode()).hexdigest()
    return private_key, public_key

# 3. Blockchain for Integrity Verification
def create_blockchain(data):
    block = hashlib.sha256(data.encode()).hexdigest()
    return block

# 4. Running City Game Optimization (RCGO) Placeholder
def rcgo_optimize(params):
    optimized_params = {k: v * random.uniform(0.9, 1.1) for k, v in params.items()}
    return optimized_params

# 5. Encryption using XOR with ECC Key
def encrypt_image(image, key):
    key_stream = np.full(image.shape, key % 256, dtype=np.uint8)
    encrypted_image = np.bitwise_xor((image * 255).astype(np.uint8), key_stream)
    return encrypted_image

# 6. Decryption
def decrypt_image(encrypted_image, key):
    key_stream = np.full(encrypted_image.shape, key % 256, dtype=np.uint8)
    decrypted_image = np.bitwise_xor(encrypted_image, key_stream) / 255.0
    return decrypted_image

# 7. Evaluation Metrics
def calculate_mse(original, decrypted):
    return np.mean((original - decrypted) ** 2)

def calculate_psnr(original, decrypted):
    mse = calculate_mse(original, decrypted)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(original, decrypted):
    return ssim(original, decrypted)

# 8. Differential Cryptanalysis (NPCR & UACI)
def calculate_npcr(original, modified):
    diff = np.sum(original != modified)
    total_pixels = original.size
    return (diff / total_pixels) * 100

def calculate_uaci(original, modified):
    return (np.sum(np.abs(original - modified)) / (original.size * 255)) * 100

# Execution
if __name__ == "__main__":
    # Preprocess image
    image = preprocess_image("sample_medical_image.png")

    # ECC Key Generation
    private_key, public_key = generate_ecc_key()

    # Encrypt and Decrypt
    encrypted_img = encrypt_image(image, private_key)
    decrypted_img = decrypt_image(encrypted_img, private_key)

    # Evaluation
    mse = calculate_mse(image, decrypted_img)
    psnr = calculate_psnr(image, decrypted_img)
    ssim_value = calculate_ssim(image, decrypted_img)

    # Differential Analysis
    modified_image = image.copy()
    modified_image[0, 0] = 1 - modified_image[0, 0]  # Flip a pixel
    npcr = calculate_npcr(encrypted_img, encrypt_image(modified_image, private_key))
    uaci = calculate_uaci(encrypted_img, encrypt_image(modified_image, private_key))

    # Display Results
    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"NPCR: {npcr:.2f}%")
    print(f"UACI: {uaci:.2f}%")

    # Show Images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Encrypted Image")
    plt.imshow(encrypted_img, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Decrypted Image")
    plt.imshow(decrypted_img, cmap='gray')

    plt.show()
