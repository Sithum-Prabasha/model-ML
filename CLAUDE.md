# Project guide (for Claude Code running on this laptop)

Coursework: PUSL3123 AI and Machine Learning, Assessment 1. CIFAR-10 with PyTorch, four notebooks
(`Ex1_MLP`, `Ex2_CNN`, `Ex3_Regularisation`, `Ex4_Width_Depth`). The notebooks are written elsewhere and
dropped into this folder; your job here is mainly to **set up the environment and run them**.

## Setup (Windows)
1. Check Python (3.10+) and whether an NVIDIA GPU exists (`nvidia-smi`).
2. `python -m venv .venv` then `.venv\Scripts\activate`, then `pip install -r requirements.txt`.
3. If there is an NVIDIA GPU, replace the default torch with the CUDA build from https://pytorch.org/get-started/locally/
   and confirm `python -c "import torch; print(torch.cuda.is_available())"` prints True.

## Running a notebook
- Smoke test first (2 epochs): set `QUICK_RUN=1` (PowerShell: `$env:QUICK_RUN="1"`), run, then unset it.
- Real run, executed in place so outputs are saved inside the notebook:
  `jupyter nbconvert --to notebook --execute --inplace Ex1_MLP.ipynb --ExecutePreprocessor.timeout=-1`
- Run from the repository root so `data/` and `results/` are created here.
- When a run finishes, tell the user the key numbers from `results/*.json` (train / val / test accuracy, best epoch,
  training time) and mention any warnings or errors with the full traceback.

## Rules
- Do not change the seed, the validation split, the hyper-parameters or the model definitions unless the user asks.
- The CIFAR-10 test set must never be used for training, tuning or model selection.
- Do not edit `results/*.json` by hand. Do not commit `data/` or `.venv/`.
- Keep executed outputs in the notebooks: the marker needs to see them.
