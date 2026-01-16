# ST-NFE: Spatiotemporal Neural Field Encoder 🧠✨

**Neurodynamic-Informed EEG-to-Text Decoding via Spatiotemporal Neural Fields**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

## 📖 Introduction (简介)

This project implements the **ST-NFE (Spatiotemporal Neural Field Encoder)** framework. It bridges brain dynamics and semantic decoding by fusing temporal causal perturbations with spatial source localization.

**Core Components:**
1.  **NPI (Neural Perturbation Inference):** Captures temporal causal flow via simulated perturbations on HBN-EEG data.
2.  **Spatial DNN:** Reconstructs source space structure from HCP-MRI data.
3.  **Neural Field Encoder:** Fuses spatiotemporal features via Posterior Learning (KL Loss).
4.  **MAML Meta-Learning:** Adapts to few-shot EEG-to-Text tasks on the ChineseEEG dataset.

## 🚀 Quick Start (快速开始)

### 1. Installation
```bash
pip install -r requirements.txt
