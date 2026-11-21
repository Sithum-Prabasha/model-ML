# PUSL3123 AI and Machine Learning - Assessment 1 (Set Exercises)

CIFAR-10 image classification with PyTorch: MLP vs CNN, regularisation, and the effect of model width and depth.

## Contents

| File | Exercise |
|---|---|
| `Ex1_MLP.ipynb` | Exercise 1 - Multi-Layer Perceptron (3 trainable layers) |
| `Ex2_CNN.ipynb` | Exercise 2 - CNN and MLP vs CNN comparison |
| `Ex3_Regularisation.ipynb` | Exercise 3 - dropout and data augmentation (3 seeds per configuration) |
| `Ex4_Width_Depth.ipynb` | Exercise 4 - model width and depth (3 seeds per configuration) |
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
3. Set the environment variable `QUICK_RUN=1` for a 2-epoch smoke test. A smoke test overwrites the files in `results/`, so finish with a full run.
4. Run the notebooks in order: Exercise 2 reads the Exercise 1 results, and Exercise 4 checks its baseline against the Exercise 3 results.

Every notebook uses the same configuration: the standard CIFAR-10 partitions (50,000 train / 10,000 test), a fixed 5,000-image validation split held out from the training set, batch size 128, Adam with learning rate 1e-3, and 30 epochs. The epoch with the best **validation** accuracy is selected; the test set is used only for final reporting.

## Multi-seed protocol

Exercises 1 and 2 are single runs with seed 42. Exercises 3 and 4 train every configuration with three run seeds (42, 43, 44):

- `SPLIT_SEED = 42` fixes the train/validation split, so it is identical for every experiment and every run.
- The run seed controls weight initialisation, dropout and augmentation randomness, and the shuffling of the training batches.
- Results are reported as mean ± sample standard deviation (`ddof=1`) over the three seeds. Exercise 3 also reports paired differences against the baseline run with the same seed.
- The dropout rate in Exercise 3 is chosen by mean validation accuracy. The test set is not used for this choice.

## Extensions (outside the main protocol)

Each extension is a clearly labelled final section of its notebook, with its own results file. The main results and their JSON files remain the reported results. Re-running a whole notebook also re-executes its main sections and rewrites their JSON file: the metrics reproduce on the same hardware, but the recorded training times change.

- **Exercise 3, Section 11 - longer training:** the four main methods trained for 60 epochs instead of 30, with the same seeds and settings (`results/ex3_extension_60_epochs.json`).
