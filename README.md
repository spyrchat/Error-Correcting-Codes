# Error-Correction-Codes Project

This repository contains Python scripts and resources for exploring and simulating Error Correction Codes (ECC). The project focuses on coding schemes such as regular and irregular LDPC codes to reduce the Bit Error Rate (BER) and optimize communication over noisy and erasure channels.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Setup Instructions](#setup-instructions)
- [Usage Instructions](#usage-instructions)
- [File Descriptions](#file-descriptions)
- [Results and Reports](#results-and-reports)
---

## Overview

This project includes:
1. Performance analysis of linear error correction codes under varying Signal-to-Noise Ratios (SNR).
2. Evaluating erasure probabilities using regular LDPC codes.
3. Designing and analyzing irregular LDPC codes.
4. Simulating erasure-erasure wiretap channels to study secure communication.

---

## Features

- Simulates regular and irregular LDPC codes.
- Examines BER vs SNR behavior.
- Models noisy and erasure communication channels.
- Implements CUDA-accelerated decoding for efficiency.

---

## Dependencies

To run this project, install the following:
- Python 3.8 or later
- `numpy`
- `scipy`
- `matplotlib`
- `torch` (for CUDA-based decoding)
- `cupy` (for CUDA-based decoding)
- `pytest` (for testing)

Install all dependencies using:
```bash
pip install numpy scipy matplotlib torch pytest cupy
