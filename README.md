# 🚀 Transformer from Scratch (PyTorch)

> 📖 Implementation of the paper *Attention Is All You Need* (Vaswani et al., 2017)  
> [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

This project is a **from-scratch PyTorch implementation** of the **Transformer architecture**, faithfully following the original paper.  
It includes **Encoder + Decoder stacks**, **Multi-Head Attention**, **Positional Encodings**, and a **training pipeline** for experimentation.


![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-1.24-orange?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Highlights
- 🧠 **Full Transformer** (Encoder + Decoder) as described in the paper  
- 🎯 **Multi-Head Attention** & **Scaled Dot-Product Attention**  
- 📐 **Sinusoidal Positional Encoding**  
- 🔄 **Residual Connections + Layer Normalization**  
- 📈 **Warmup Learning Rate Schedule** (Noam LR)  
- 🧪 **Toy Copy Task** for quick verification  
- ⚡ Modular codebase — easy to extend for translation, summarization, etc.  

---

## 📖 Paper Reference
📑 **Attention Is All You Need**  
*Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, Illia Polosukhin*  
Published at NeurIPS 2017.  
[👉 Read on arXiv](https://arxiv.org/abs/1706.03762)

---

## 🏗️ Project Structure
```bash
transformer-from-scratch/
├── data/ # placeholder for datasets
├── model/
│ ├── init.py
│ ├── layers.py # Scaled Dot-Product, MultiHeadAttention, FFN, EncoderLayer, DecoderLayer
│ ├── transformer.py # Encoder, Decoder, Full Transformer model
│ ├── utils.py # Masks, Learning Rate Scheduler
├── train.py # Training loop (toy copy task)
├── evaluate.py # Greedy decoding
├── requirements.txt # Dependencies
└── README.md 
```
---

## ⚙️ Installation

```bash
### Clone the repo
git clone https://github.com/YOUR_USERNAME/transformer-from-scratch.git
cd transformer-from-scratch

### Create conda environment
conda create -n transformer python=3.10 -y
conda activate transformer

### Install dependencies
pip install -r requirements.txt
```
---
## 🏃 Training (Toy Copy Task)

We start with a simple copy task: the model learns to output the same sequence it receives.
```bash
python train.py
```
---
## 📌 Example training log:

Epoch 1: Loss = 4.01
Epoch 2: Loss = 3.89
Epoch 3: Loss = 3.87
...

✅ Loss decreases → confirms the Transformer is working.

---

## 🔍 Evaluation (Greedy Decode)
```bash
python evaluate.py
```
Example:
Input  : [1, 23, 45, 12]
Output : [1, 23, 45, 12]

---
## 📊 Roadmap
Add Label Smoothing (ε = 0.1) as in the paper
Train on real translation datasets (e.g., WMT14 En↔De)
Implement Beam Search decoding (beam size = 4)
Scale up to larger models (d_model=512, 6 layers)
Run on GPU/TPU (Kaggle, Colab, or cloud)

---
## 🛠️ Tech Stack
1. PyTorch
2. NumPy
3. tqdm
4. einops

---
