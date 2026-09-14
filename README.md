# Deep Learning Lab (PyTorch)

A hands-on lab notebook covering PyTorch fundamentals — from raw tensors to trained neural networks — built and run in Google Colab (GPU: T4).

## What's inside

**1. Tensor basics**
- Creating tensors (`empty`, `zeros`, `ones`, `rand`, `randint`)
- Element-wise ops, reductions (`sum`, `prod`), matrix multiplication
- NumPy interop (`from_numpy`, `.numpy()`)
- A small tensor-slicing exercise (drawing a pixel-art face on a 10x10 grid)

**2. Activation functions & gradients**
- Step, Sigmoid, and Tanh — plotted alongside their autograd-computed gradients
- ReLU, Leaky ReLU, and ELU — same treatment
- A guided exercise probing `torch.autograd.grad` on scalar activations

**3. Perceptron / single linear unit**
- Manual weight assignment on `nn.Linear` to replicate a simple logic gate

**4. Logistic Regression — Breast Cancer Wisconsin dataset**
- Custom `MySimpleNN` (`nn.Module`) with a linear layer + sigmoid
- Preprocessing with `StandardScaler` / `LabelEncoder`, train/test split

**5. MLP Classifier — Social Network Ads dataset**
- `nn.Sequential` model (2 → 10 → 10 → 1) with ReLU activations
- Custom training loop with mini-batch SGD (`build_model` / `fit`)

**6. Deep Neural Network — Fashion-MNIST**
- `MyNN` model (784 → 128 → 64 → 10) trained with `DataLoader`/`TensorDataset`
- Train vs. test accuracy comparison, with a baseline (no regularization) run

## Tech stack
- Python, PyTorch, NumPy, pandas, scikit-learn, Matplotlib
- Originally developed in Google Colab

## Getting started

```bash
git clone https://github.com/<your-username>/deep-learning-lab-pytorch.git
cd deep-learning-lab-pytorch
pip install torch numpy pandas scikit-learn matplotlib jupyter
jupyter notebook Deep_Learning_Lab.ipynb
```

The notebook pulls its datasets directly from public URLs (Breast Cancer, Social Network Ads, Fashion-MNIST), so no manual data download is required — just run the cells top to bottom.

## Notes
- A GPU (e.g. Colab's T4) speeds up the Fashion-MNIST section but isn't required; everything falls back to CPU automatically (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).
- Some cells are left as guided TODO exercises (e.g. manual MAE loss implementation) — fill those in as practice.

## License
Add a license of your choice (e.g. MIT) if you plan to share this publicly.
