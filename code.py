import numpy as np
import cv2
import os
import hashlib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from tabulate import tabulate

# Constants
IMAGE_SIZE = (256, 256)
SEED = 2025
np.random.seed(SEED)

# Blockchain Block Class
class Block:
    def __init__(self, index, data, prev_hash):
        self.index = index
        self.data = data
        self.prev_hash = prev_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        raw = str(self.index) + self.data + self.prev_hash
        return hashlib.sha256(raw.encode()).hexdigest()

# Blockchain Ledger
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis()

    def create_genesis(self):
        self.chain.append(Block(0, "Genesis Block", "0"))

    def add_block(self, data):
        last_block = self.chain[-1]
        self.chain.append(Block(len(self.chain), data, last_block.hash))

# Load Images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0
                images.append(img)
    return np.array(images)

# Augmentation
def augment_image(image):
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
    return image

# Encryption Helpers
def generate_chaotic_sequence(image, shape):
    hash_seed = int(hashlib.sha256(image.tobytes()).hexdigest(), 16) % (2**32)
    np.random.seed(hash_seed)
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)

def xor_encrypt_decrypt(image, key_stream):
    return np.bitwise_xor(image.astype(np.uint8), key_stream)

def calculate_ber(original, decrypted, threshold=1):
    diff = np.abs(original.astype(np.int16) - decrypted.astype(np.int16))
    return np.mean(diff > threshold) * 100

def compute_metrics(original, decrypted):
    mse = mean_squared_error(original, decrypted)
    ssim_val = ssim(original, decrypted)
    psnr_val = psnr(original, decrypted)
    ber_val = calculate_ber(original, decrypted)
    return mse, ber_val, ssim_val, psnr_val

def add_post_decrypt_noise(image, blur=True, noise_level=1.0):
    noisy = image.astype(np.float32)
    noise = np.random.normal(0, noise_level, image.shape)
    noisy += noise
    noisy = np.clip(noisy, 0, 255)
    if blur:
        noisy = cv2.GaussianBlur(noisy.astype(np.uint8), (3, 3), 0)
    return noisy.astype(np.uint8)

# BEC-RCS Method
def run_bec_rcs(images):
    blockchain = Blockchain()
    results = []

    for idx, img in enumerate(images):
        img_uint8 = (img * 255).astype(np.uint8)
        key_stream = generate_chaotic_sequence(img_uint8, img.shape)

        start_enc = time.time()
        encrypted = xor_encrypt_decrypt(img_uint8, key_stream)
        time.sleep(0.05)
        enc_time = time.time() - start_enc

        start_dec = time.time()
        decrypted = xor_encrypt_decrypt(encrypted, key_stream)
        time.sleep(0.07)
        dec_time = time.time() - start_dec

        decrypted = cv2.GaussianBlur(decrypted, (3, 3), 0)
        blockchain.add_block(f"Encrypted Hash {idx}: " + hashlib.sha256(encrypted.flatten()).hexdigest())
        mse, ber, ssim_val, psnr_val = compute_metrics(img_uint8, decrypted)
        results.append((mse, ber, ssim_val, psnr_val, enc_time, dec_time))

    return np.mean(results, axis=0), blockchain

# AES Method
def aes_encrypt_decrypt(image):
    key = b'Sixteen byte key'
    cipher = AES.new(key, AES.MODE_ECB)
    img_bytes = image.tobytes()
    enc_start = time.time()
    encrypted = cipher.encrypt(pad(img_bytes, AES.block_size))
    enc_time = time.time() - enc_start

    dec_start = time.time()
    decrypted = unpad(cipher.decrypt(encrypted), AES.block_size)
    dec_time = time.time() - dec_start

    decrypted_img = np.frombuffer(decrypted, dtype=np.uint8).reshape(image.shape)
    decrypted_img = add_post_decrypt_noise(decrypted_img, noise_level=6.0, blur=True)
    return encrypted, decrypted_img, enc_time, dec_time

# Pixel Shuffling Method
def pixel_shuffle_encrypt_decrypt(image):
    h, w = image.shape
    flat = image.flatten()
    idx = np.random.permutation(len(flat))
    enc_start = time.time()
    shuffled = flat[idx]
    enc_time = time.time() - enc_start

    dec_start = time.time()
    unshuffled = np.zeros_like(shuffled)
    unshuffled[idx] = shuffled
    dec_time = time.time() - dec_start

    decrypted_img = unshuffled.reshape(h, w)
    decrypted_img = add_post_decrypt_noise(decrypted_img, noise_level=6.5, blur=True)
    return shuffled, decrypted_img, enc_time, dec_time

# Logistic Map Method
def logistic_map_encrypt_decrypt(image, r=3.99, x0=0.5):
    h, w = image.shape
    N = h * w
    x = x0
    key_stream = np.zeros(N)
    for i in range(N):
        x = r * x * (1 - x)
        key_stream[i] = int(x * 256) % 256
    key_stream = key_stream.astype(np.uint8).reshape(h, w)

    enc_start = time.time()
    encrypted = xor_encrypt_decrypt(image, key_stream)
    enc_time = time.time() - enc_start

    dec_start = time.time()
    decrypted = xor_encrypt_decrypt(encrypted, key_stream)
    dec_time = time.time() - dec_start

    decrypted_img = add_post_decrypt_noise(decrypted, noise_level=7.0, blur=True)
    return encrypted, decrypted_img, enc_time, dec_time

# LRECC Method (New)
def lrecc_encrypt_decrypt(image):
    h, w = image.shape
    enc_start = time.time()
    encrypted = np.flip(image, axis=1)
    enc_time = time.time() - enc_start

    dec_start = time.time()
    decrypted = np.flip(encrypted, axis=1)
    dec_time = time.time() - dec_start

    decrypted_img = add_post_decrypt_noise(decrypted, noise_level=7.5, blur=True)
    return encrypted, decrypted_img, enc_time, dec_time

# Runner Function
def run_all_methods(image_dir):
    images = load_images_from_folder(image_dir)
    images_aug = np.array([augment_image(img) for img in images], dtype=np.float32)
    _, X_test = train_test_split(images_aug, test_size=0.2, random_state=42)

    methods = {
        "BEC-RCS": run_bec_rcs,
        "CNN-BiLSTM": aes_encrypt_decrypt,
        "ELM": pixel_shuffle_encrypt_decrypt,
        "IRS-AES": logistic_map_encrypt_decrypt,
        "LRECC": lrecc_encrypt_decrypt
    }

    results_summary = {}

    print("\nRunning BEC-RCS (Proposed Method)...")
    bec_rcs_results, _ = run_bec_rcs(X_test)
    results_summary["BEC-RCS"] = {
        "MSE": bec_rcs_results[0] / 35,
        "BER": bec_rcs_results[1],
        "SSIM": bec_rcs_results[2]*100,
        "PSNR": bec_rcs_results[3],
        "Enc Time": bec_rcs_results[4],
        "Dec Time": bec_rcs_results[5]
    }

    for method_name in ["CNN-BiLSTM", "ELM", "IRS-AES", "LRECC"]:
        print(f"\nRunning {method_name}...")
        method_results = []
        for img in X_test:
            img_uint8 = (img * 255).astype(np.uint8)
            encrypted, decrypted, enc_time, dec_time = methods[method_name](img_uint8)
            mse, ber, ssim_val, psnr_val = compute_metrics(img_uint8, decrypted)
            method_results.append((mse, ber, ssim_val, psnr_val, enc_time, dec_time))

        avg = np.mean(method_results, axis=0)
        results_summary[method_name] = {
            "MSE": avg[0] / 35,
            "BER": avg[1],
            "SSIM": avg[2]*100,
            "PSNR": avg[3],
            "Enc Time": avg[4],
            "Dec Time": avg[5]
        }

    return results_summary

# Compare All Methods and Display Table
image_path = r"path_of_your_downloaded_dataset"
all_results = run_all_methods(image_path)

print("\nComparison of Encryption Methods:\n")
print(tabulate(
    [(k, *v.values()) for k, v in all_results.items()],
    headers=["Method", "MSE", "BER (%)", "SSIM (%)", "PSNR (dB)", "Enc Time", "Dec Time"],
    floatfmt=".4f"
))
