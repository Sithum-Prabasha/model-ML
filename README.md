# PUSL3123 AI and Machine Learning - Assessment 1 (Set Exercises)

CIFAR-10 image classification with PyTorch: MLP vs CNN, regularisation, and the effect of model width and depth.

## Contents

Run the notebooks in this order. Exercise 2 reads the Exercise 1 results, and Exercise 4 checks its baseline against the Exercise 3 results.

| Order | File | What it does | Measured runtime (GPU) |
|---|---|---|---|
| 1 | `Ex1_MLP.ipynb` | Trains and evaluates a three-layer Multi-Layer Perceptron | about 40 s |
| 2 | `Ex2_CNN.ipynb` | Trains a three-layer CNN and compares it with the MLP | about 40 s |
| 3 | `Ex3_Regularisation.ipynb` | Evaluates dropout and data augmentation over 3 seeds, with a 60-epoch extension | about 24 min |
| 4 | `Ex4_Width_Depth.ipynb` | Evaluates model width and depth over 3 seeds, with a learning-rate extension | about 34 min |

Runtimes are wall-clock times for one clean run on an NVIDIA GeForce RTX 4050 Laptop GPU, including the extensions. The training times recorded in `results/*.json` come from the final clean run on the same RTX 4050 Laptop GPU and are indicative only: they vary between runs and machines, while the accuracies and losses do not.

Other files:

- `results/`: JSON results and, in `results/figures/`, the figures written by the notebooks.
- `requirements.txt`: exact package versions used.
- `report_figures/`: the figures used in the report, numbered as in the report (see `report_figures/INDEX.md`). `report_figures/make_report_figures.py` creates report Figures 2, 4, 6 and 9 from `results/*.json`: `python report_figures/make_report_figures.py results <output_folder>`.
- `tools/compare_results.py`: compares `results/*.json` with a backup copy (see below). It is kept in the project repository and is not part of the submission ZIP.

## Setup

```
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

**GPU:** an NVIDIA GPU with CUDA is recommended. The notebooks use it automatically when available. They also run on the CPU, but much more slowly, particularly Exercises 3 and 4. For a CPU-only machine, follow the comment at the top of `requirements.txt`.

**Data:** CIFAR-10 is not included. `torchvision` downloads it into `data/` the first time a notebook runs.

## Running the notebooks

Run from the project folder, so that `data/` and `results/` are created there.

1. Open a notebook in Jupyter or VS Code and choose *Run All*, or run it headless:
   `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Ex1_MLP.ipynb`
2. Set the environment variable `QUICK_RUN=1` for a 2-epoch smoke test. A smoke test overwrites the files in `results/`, so finish with a full run.

Each notebook writes its numbers to `results/` and its figures to `results/figures/`. Re-running a notebook overwrites them. On the same hardware and package versions the metrics reproduce exactly, while recorded training times vary slightly.

## Experimental protocol

Every notebook uses the standard CIFAR-10 partitions (50,000 train / 10,000 test), a fixed 5,000-image validation split held out from the training set, batch size 128, Adam with learning rate 1e-3, cross-entropy loss and 30 epochs. The epoch with the best **validation** accuracy is restored; the test set is used only for final reporting.

Exercises 1 and 2 are single runs with seed 42. Exercises 3 and 4 train every configuration with three run seeds (42, 43, 44):

- `SPLIT_SEED = 42` fixes the train/validation split, so it is identical for every experiment and every run.
- The run seed controls weight initialisation, dropout and augmentation randomness, and the shuffling of the training batches.
- Results are reported as mean ± sample standard deviation (`ddof=1`) over the three seeds. Exercise 3 also reports paired differences against the baseline run with the same seed.
- The dropout rate in Exercise 3 is chosen by mean validation accuracy. The test set is not used for this choice.

## Extensions (outside the main protocol)

Each extension is a clearly labelled final section of its notebook, with its own results file. The main results remain the reported results.

- **Exercise 3, Section 11 - longer training:** the four main methods trained for 60 epochs instead of 30, with the same seeds and settings (`results/ex3_extension_60_epochs.json`).
- **Exercise 4, Section 11 - learning rate for wider models:** 32-64, 64-128 and 128-256 trained at learning rate 3e-4 and compared with the saved 1e-3 results, judged on validation accuracy (`results/ex4_extension_lr.json`).

## Checking reproducibility

`tools/compare_results.py` compares every JSON file in `results/` with a backup folder, value by value: accuracies, losses, selected epochs, summaries, paired differences, parameter counts and MACs. It ignores only training-time fields, and it lists any difference.

```
python tools/compare_results.py --new results --old ../results_backup_before_final_run
```
