# BEC-RCS: Blockchain Elliptic Curve Cryptography-Based Running City Game Search for Medical Image Security

## 📖 Overview
The **BEC-RCS** model integrates **Blockchain**, **Elliptic Curve Cryptography (ECC)**, and the **Running City Game Optimization (RCGO)** algorithm to provide robust security for medical images. This repository includes the source code for encryption, decryption, attack analysis, and evaluation metrics.

## 📂 Project Structure
```
BEC-RCS/
│
├── main.py               # Main implementation file
├── sample_medical_image.png # Sample medical image for testing
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── results/              # Output images and analysis
```

## ⚙️ Features
- **Preprocessing**: Resizing, Normalization, and Augmentation
- **ECC-Based Encryption/Decryption**: Ensures data confidentiality
- **Blockchain Integration**: Guarantees data integrity
- **Running City Game Optimization (RCGO)**: Hyperparameter tuning
- **Differential Cryptanalysis**: NPCR and UACI metrics for attack analysis

## 📊 Evaluation Metrics
- **MSE (Mean Squared Error)**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index Measure)**
- **NPCR (Number of Pixel Change Rate)**
- **UACI (Unified Average Changing Intensity)**

## 📥 Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/BEC-RCS.git
   cd BEC-RCS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🧮 Usage
1. **Preprocess and Encrypt Image:**
   ```bash
   python main.py
   ```

2. **Evaluate Metrics:**
   The script outputs **MSE**, **PSNR**, **SSIM**, **NPCR**, and **UACI** values, along with visualizations of the encrypted and decrypted images.

3. **Run Differential Cryptanalysis:**
   The script performs horizontal, vertical, and diagonal attacks and outputs corresponding NPCR and UACI metrics.

## 🖼️ Sample Output
- **Original Image**
- **Encrypted Image**
- **Decrypted Image**
- **Attack Analysis Metrics**

## 🔐 Security Analysis
This project evaluates the model's resistance against differential cryptanalysis and validates its robustness using NPCR and UACI under various attack conditions.

## 💡 Future Enhancements
- Implement real-time encryption in IoT healthcare devices.
- Optimize for high-resolution medical images.
- Extend security analysis with additional cryptographic attacks.
