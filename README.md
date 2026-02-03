# 🌙 Urdu to Roman Urdu Neural Machine Translation

A deep learning-based Neural Machine Translation (NMT) system that translates Urdu script to Roman Urdu (Romanized Urdu). Built with PyTorch and featuring a beautiful Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [Experiments](#experiments)
- [Author](#author)

## 🔍 Overview

This project implements a **Sequence-to-Sequence (Seq2Seq)** model with a **Bidirectional LSTM Encoder** and **LSTM Decoder** for translating Urdu text (in Nastaliq script) to Roman Urdu. The model is trained on a curated dataset of Urdu poetry from renowned poets.

### Key Features

- **BiLSTM Encoder**: Captures bidirectional context from Urdu input
- **Character-level Tokenization**: Handles the complexity of Urdu script at the character level
- **Teacher Forcing**: Improves training convergence
- **Multiple Experiments**: Compare different hyperparameter configurations
- **Beautiful Web UI**: Interactive Streamlit interface with glassmorphism design

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Seq2Seq Model                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐     ┌──────────────────────┐         │
│  │    Encoder (BiLSTM)  │     │    Decoder (LSTM)    │         │
│  ├──────────────────────┤     ├──────────────────────┤         │
│  │ • Embedding Layer    │     │ • Embedding Layer    │         │
│  │ • 2-Layer BiLSTM     │────▶│ • 4-Layer LSTM       │         │
│  │ • Dropout (0.3)      │     │ • Fully Connected    │         │
│  │ • Hidden Dim: 512    │     │ • Dropout (0.3)      │         │
│  └──────────────────────┘     └──────────────────────┘         │
│                                                                 │
│  Input: Urdu Script              Output: Roman Urdu             │
│  "دل سے اتر جائے گا"       ──────▶    "dil se utar jaae ga"    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Component | Specification |
|-----------|---------------|
| Encoder | Bidirectional LSTM |
| Decoder | Unidirectional LSTM |
| Embedding Dimension | 256 |
| Hidden Dimension | 512 |
| Encoder Layers | 2 |
| Decoder Layers | 4 |
| Dropout | 0.3 |

## 📚 Dataset

The dataset consists of Urdu poetry from **30 renowned poets**, including:

- Mirza Ghalib
- Allama Iqbal
- Faiz Ahmad Faiz
- Ahmad Faraz
- Parveen Shakir
- Jaun Eliya
- And many more...

Each poet's folder contains:
- `ur/` - Urdu script version of poems
- `en/` - Roman Urdu transliteration

**Total parallel pairs**: ~3900+ sentence pairs

## 📁 Project Structure

```
NLP/
├── app.py                    # Streamlit web application
├── src/
│   ├── data_loader.py        # Dataset loading & tokenization
│   ├── model.py              # Seq2Seq model architecture
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation metrics (BLEU, CER)
├── checkpoints/
│   ├── exp1_baseline_best.pt       # Baseline model weights
│   ├── exp2_small_hidden_best.pt   # Small hidden dim model
│   └── exp3_high_dropout_best.pt   # High dropout model
├── dataset/                  # Poetry dataset by poet
├── test_data_loader.py       # Data loading tests
├── test_model.py             # Model tests
├── verify_inference.py       # Inference verification script
└── README.md
```

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/Haseebabbas2/Urdu-to-Roman-Urdu-Translator.git
cd Urdu-to-Roman-Urdu-Translator

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch streamlit
```

## 🚀 Usage

### Web Application

Launch the interactive Streamlit interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and start translating!

### Programmatic Usage

```python
from src.model import Encoder, Decoder, Seq2Seq
from src.data_loader import Tokenizer, load_data
import torch

# Load tokenizers
pairs = load_data('dataset')
src_tokenizer, tgt_tokenizer = Tokenizer(), Tokenizer()
src_tokenizer.build_vocab([p[0] for p in pairs])
tgt_tokenizer.build_vocab([p[1] for p in pairs])

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc = Encoder(src_tokenizer.vocab_size, 256, 512, 2, 0.3)
dec = Decoder(tgt_tokenizer.vocab_size, 256, 512, 4, 0.3)
model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('checkpoints/exp1_baseline_best.pt'))
model.eval()
```

### Verify Inference

Run the verification script to test translations:

```bash
python verify_inference.py
```

## 🎓 Training

To train the model from scratch:

```bash
python -m src.train
```

This runs three experiments with different configurations:
1. **Baseline**: Standard configuration
2. **Small Hidden**: Reduced hidden dimension (256)
3. **High Dropout**: Increased dropout (0.5)

### Training Configuration

```python
config = {
    'emb_dim': 256,
    'hid_dim': 512,
    'enc_layers': 2,
    'dec_layers': 4,
    'dropout': 0.3,
    'lr': 1e-3,
    'batch_size': 128,
    'epochs': 5,
    'clip': 1
}
```

## 📊 Evaluation Metrics

The model is evaluated using:

| Metric | Description |
|--------|-------------|
| **BLEU** | Bilingual Evaluation Understudy score (n-gram precision) |
| **CER** | Character Error Rate (Levenshtein distance based) |
| **Perplexity** | Model uncertainty measure (lower is better) |

## 🧪 Experiments

| Experiment | Hidden Dim | Dropout | Description |
|------------|------------|---------|-------------|
| Baseline | 512 | 0.3 | Standard configuration |
| Small Hidden | 256 | 0.3 | Reduced model size |
| High Dropout | 512 | 0.5 | More regularization |

Pre-trained checkpoints for all experiments are available in the `checkpoints/` directory.

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **Streamlit** - Web application framework
- **CUDA/MPS** - GPU acceleration support

## 👤 Author

**Haseeb Abbas**

---

<p align="center">
  <i>Project powered by PyTorch & Streamlit</i>
</p>
