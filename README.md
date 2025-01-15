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
```

## File Descriptions

| File Name                       | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `README.md`                     | Project documentation.                                                     |
| `ergasia 2023-2024.pdf`         | Project guidelines and tasks.                                              |
| `make_ldpc.py`                  | Generates regular LDPC matrices.                                           |
| `encoder.py`                    | Encodes data using LDPC.                                                   |
| `decoder.py`                    | Decodes data using LDPC (CPU-based).                                       |
| `decoder_cuda.py`               | CUDA-accelerated LDPC decoding.                                            |
| `construct_irregular_ldpc.py`   | Generates irregular LDPC matrices.                                         |
| `erasure_channel_encoding.py`   | Simulates erasure channels with regular LDPC codes.                        |
| `erasure_channel_encoding_irregular.py` | Simulates erasure channels with irregular LDPC codes.             |
| `bpsk.py`                       | BPSK modulation and decoding.                                              |
| `bpsk_hamming.py`               | BPSK with Hamming code simulation.                                         |
| `test_irregular_ldpc.py`        | Unit tests for irregular LDPC.                                             |
| `simulation_ex2.py`             | Additional simulations for experiment 2.                                   |
| `demo.py`                       | Complete system demonstration.                                             |
