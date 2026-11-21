# PUSL3123 AI and Machine Learning - Assessment 1 (Set Exercises)

CIFAR-10 image classification with PyTorch: MLP vs CNN, regularisation, and the effect of model width and depth.

## Contents

| File | Exercise |
|---|---|
| `Ex1_MLP.ipynb` | Exercise 1 - Multi-Layer Perceptron (3 trainable layers) |
| `Ex2_CNN.ipynb` | Exercise 2 - CNN and MLP vs CNN comparison |
| `Ex3_Regularisation.ipynb` | Exercise 3 - dropout and data augmentation *(to be added)* |
| `Ex4_Width_Depth.ipynb` | Exercise 4 - model width and depth *(to be added)* |
| `results/` | JSON results and figures written by the notebooks (used in the report) |
| `requirements.txt` | Python dependencies |

## Setup

```
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch from <https://pytorch.org/get-started/locally/> instead of the default wheel; the notebooks use the GPU automatically when one is available.

## Reproducing the results

1. Open a notebook in Jupyter / VS Code and choose *Run All* (CIFAR-10 downloads automatically into `data/`).
2. Or run headless: `jupyter nbconvert --to notebook --execute --inplace Ex1_MLP.ipynb`
3. Set the environment variable `QUICK_RUN=1` for a 2-epoch smoke test.

Every notebook uses the same configuration: seed 42, the standard CIFAR-10 partitions (50,000 train / 10,000 test), a fixed 5,000-image validation split held out from the training set, batch size 128, Adam with learning rate 1e-3, and up to 30 epochs. The epoch with the best **validation** accuracy is selected; the test set is used only for final reporting.
