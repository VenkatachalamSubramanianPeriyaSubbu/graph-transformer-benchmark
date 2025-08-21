````markdown
# Graph Transformer Benchmark

This repository provides an **easy-to-use implementation of Graph Transformer models** for graph classification tasks. It includes example usage with a standard benchmark dataset like **MUTAG**.


---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/graph-transformer-benchmark.git
cd graph-transformer-benchmark
````

2. Create a new Python environment (recommended):

```bash
conda create -n gst_env python=3.10
conda activate gst_env
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** Make sure you have `torch`, `torchvision`, `torchaudio`, `torch-geometric`, and `graph-transformer-pytorch` installed.

---

## Usage

### Training on MUTAG

```bash
python train_mutag.py
```

* The script automatically downloads the MUTAG dataset.
* Graphs are batched with padding and masks.
* Trains the GraphTransformer and evaluates test accuracy.

### Training on PROTEINS

```bash
python train_proteins.py
```

* Requires the PROTEINS dataset from PyTorch Geometric TUDataset.
* Works similarly to MUTAG, with automatic padding and graph pooling.

---

## File Structure

```
graph-transformer-benchmark/
│
├── train_mutag.py          # Training script for MUTAG
├── train_proteins.py       # Training script for PROTEINS
├── graph_transformer.py    # Model definition
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── data/                   # Downloaded datasets
```

---

## Example Output

```
Number of graphs: 188
Number of classes: 2
Node features shape: torch.Size([17, 7])
Edge index shape: torch.Size([2, 38])
Label: tensor([1])
...
Test Accuracy: 0.90
```

---

## Credits

The base graph transformer model is created by `lucidrains`
https://github.com/lucidrains/graph-transformer-pytorch/tree/main

---

## References

* [Graph Transformer PyTorch](https://github.com/lucidrains/graph-transformer-pytorch)
* [TUDatasets for Graph Classification](https://chrsmrrs.com/graphkerneldatasets)