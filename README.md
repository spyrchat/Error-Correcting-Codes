# 📡 Error-Correcting-Codes Project  

This repository contains Python scripts and resources for exploring and simulating **Error Correction Codes (ECC)**, with a primary focus on **Low-Density Parity-Check (LDPC) codes** for improving communication reliability over **noisy and erasure channels**.  

The project implements **both regular and irregular LDPC codes**, optimizes decoding performance using **CUDA acceleration**, and analyzes **Bit Error Rate (BER) vs. Signal-to-Noise Ratio (SNR)** characteristics.  

---

## 📖 Table of Contents  
- [🔍 Overview](#-overview)  
- [🚀 Features](#-features)  
- [📦 Dependencies](#-dependencies)  
- [⚙️ Setup Instructions](#-setup-instructions)  
- [💡 Usage Instructions](#-usage-instructions)  
- [📂 File Descriptions](#-file-descriptions)  

---

## 🔍 Overview  

This project covers:  
✅ **Performance analysis of LDPC codes** under varying **Signal-to-Noise Ratios (SNR)**.  
✅ **Erasure probability evaluation** using **regular LDPC codes**.  
✅ **Design and analysis of irregular LDPC codes**.  
✅ **CUDA-accelerated decoding** for efficiency in large-scale simulations.  

---

## 🚀 Features  

- ✅ **Simulation of Regular & Irregular LDPC Codes**  
- ✅ **BER vs. SNR Analysis** for communication performance evaluation  
- ✅ **Noisy and Erasure Channel Modeling**  
- ✅ **CUDA-Accelerated Decoding** for improved performance  

---

## 📦 Dependencies  

Ensure you have **Python 3.8+** installed. Then, install the required dependencies:  

```bash
pip install numpy scipy matplotlib torch pytest cupy-cuda12x
```

## 📂 File Descriptions  

| **File Name**                             | **Description**                                                          |
|-------------------------------------------|--------------------------------------------------------------------------|
| `README.md`                               | Project documentation.                                                   |
| `make_ldpc.py`                            | Generates **regular LDPC** parity-check matrices.                        |
| `construct_irregular_ldpc.py`             | Generates **irregular LDPC** matrices.                                   |
| `encoder.py`                              | Encodes data using **LDPC codes**.                                       |
| `decoder.py`                              | **CPU-based** LDPC decoder implementation.                               |
| `decoder_cuda.py`                         | **CUDA-accelerated** LDPC decoder.                                       |
| `erasure_channel_encoding.py`             | Simulates **erasure channels** with regular LDPC.                        |
| `erasure_channel_encoding_irregular.py`   | Simulates erasure channels with **irregular LDPC**.                      |
| `bpsk.py`                                 | **BPSK modulation** and decoding.                                        |
| `bpsk_hamming.py`                         | BPSK modulation with **Hamming Code**.                                   |
| `simulation_ex2.py`                       | Simulations for the regular LDPC.                                        |
| `simulation_ex3.py`                       | Simulations for the Irregular LDPC.                                      |
| `test_irregular_ldpc.py`                  | Unit tests for **irregular LDPC**.                                       |
| `demo.py`                                 | Complete **end-to-end system demonstration**.                            |

## ⚙️ Setup Instructions  

Follow these steps to set up the project on your local machine:  

### 1️⃣ Clone the Repository  
First, download the repository using `git clone`:  

```bash
git clone https://github.com/spyrchat/Error-Correcting-Codes.git
cd Error-Correcting-Codes

