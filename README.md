# TinyViT on CIFAR-10

An end-to-end implementation of a **Tiny Vision Transformer (TinyViT)** trained from scratch on the CIFAR-10 dataset. This repository is meant to demystify Vision Transformers by breaking down their components clearly and providing a fully functional training + demo pipeline ipynb.

---

## 🔍 Overview

Vision Transformers (ViTs) apply transformer-based architectures to image data by converting images into sequences of patch embeddings, then adds positional embeddings ready to be passed into a transformer for self attention. This project walks through a minimalist ViT architecture with detailed modular design, intuitive explanations, and self-attention visualizations (coming soon).

---

## 🧠 Architecture Breakdown

### 1. **Patch Embedding**

```python
self.patches_embedding = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=4)
```

- Input image: `32x32x3`
- Patch size: `4x4` => total patches: `(32/4)^2 = 64`
- After conv: `(B, 64, 8, 8)` -> flattened to `(B, 64, 64)`
- Output shape: `(Batch, Num_Patches, Embedding_Dim)`

> Each row is a 64-dimensional embedding for a patch.

### 2. **ViT Embedding: CLS + Positional Embedding**

```python
self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
self.pos_embedding = nn.Parameter(torch.randn(1, 65, 64))
```

- `CLS` token is prepended to the sequence.
- Output shape: `(B, 65, 64)`

> 🔍 Why positional embedding?

Let’s say the embedding says:

- Patch has black & white textures = ears.
- If it's in **top-left**, it might belong to a **panda**.
- If the same patch appears in the **bottom**, it might be **another animal** or an **upside-down image**.

> 🚀 Positional embeddings help the model understand **where** the patch came from, not just **what** it contains.

### 3. **Transformer Encoder Block**

```python
nn.LayerNorm -> nn.MultiheadAttention -> Add & Norm -> MLP -> Add & Norm
```

- `Q = K = V = normalized input` => self-attention
- Multi-head attention captures relationships between patches
- MLP refines representations

Each block retains the shape: `(B, 65, 64)`

### 4. **Classification Head**

```python
self.mlp_head = nn.Sequential(
  nn.LayerNorm(64),
  nn.Linear(64, 10)
)
```

- Uses only the `[CLS]` token at index `x[:, 0]`
- Final shape: `(B, 10)` → class logits

---

## 🧪 Dataset

- CIFAR-10
- 32x32 color images across 10 classes

---

## 🧰 File Structure

```
├── models/
│   └── tinyvit.py              # Model architecture
├── utils/
│   └── tinyvit_utils.py        # Transforms, training loop
├── main.py                     # Training script
├── main_demo.ipynb             # Inference + visualization
├── requirements.txt            # Dependencies
├── README.md                   # This file
```

---

## 🛠️ Installation

```bash
git clone https://github.com/dhaaivat/TinyVit-implementation.git
cd TinyVit-implementation
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🔧 Train the model:

```bash
python main.py
```

### 🎯 Run the demo:

```bash
Open main_demo.ipynb in Jupyter/Colab
```

---

## 🔍 Attention Map Visualization (Coming Soon)

We’ll visualize how the `[CLS]` token attends to each image patch using heatmaps.

---

## 📈 Performance

| Epoch | Train Acc | Val Acc |
| ----- | --------- | ------- |
| 10    | \~63.6%   | \~60.4% |

Not state-of-the-art — but the point is **architectural and conceptual clarity**.
**NOTE:** Transformers are extremely data hungry and would outperform CNN based model for classification **if** trained on extremely large amounts of data.

---

## 💡 Inspiration

This project was designed from scratch with a goal of **fully understanding ViTs** — not just using them. Through hands-on debugging and shape-checks, this repo explains:

- Why patch tokenization matters
- What positional embeddings do
- How self-attention interacts with visual data

---

## 🧠 Author

**Dhaaivat Patil** 

---

## 📜 License

MIT License

---

## Acknowledgements:-

- Original ViT Paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- CIFAR-10 [About the dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## ⭐️ Star this repo if it helped you understand ViTs!

