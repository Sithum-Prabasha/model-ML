"""Extra report figures for PUSL3123 Assessment 1, built only from results/*.json (no retraining).

Usage: python report_figures/make_report_figures.py results <output_folder>
Written with generative-AI assistance (Claude, Anthropic); see the report's AI declaration.
"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
R = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
J = lambda f: json.load(open(f"{R}/{f}"))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, GOLD, GREY = "#2a78d6", "#eb6834", "#1baa7a", "#eda100", "#555555"
CLASSES = ["aeroplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
e1, e2 = J("ex1_mlp.json"), J("ex2_cnn.json")

# A. MLP vs CNN: training AND validation loss/accuracy (brief Ex2 req. 4)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ep = np.arange(1, 31)
for e, name, c in ((e1, "MLP", BLUE), (e2, "CNN", ORANGE)):
    h = e["history"]
    ax[0].plot(ep, h["train_loss"], "--", c=c, label=f"{name} training")
    ax[0].plot(ep, h["val_loss"], "-", c=c, label=f"{name} validation")
    ax[1].plot(ep, np.array(h["train_acc"])*100, "--", c=c, label=f"{name} training")
    ax[1].plot(ep, np.array(h["val_acc"])*100, "-", c=c, label=f"{name} validation")
    for a in ax: a.axvline(e["best_epoch"], c=c, lw=0.8, ls=":")
ax[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
ax[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, ncol=4, loc="lower center", fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("MLP vs CNN: training (dashed) and validation (solid) curves; dotted lines = selected epochs", fontsize=10)
fig.tight_layout(rect=[0, 0.06, 1, 0.94])
fig.savefig(f"{OUT}/fig_ex2_mlp_vs_cnn_train_val.png"); plt.close(fig)

# B. Row-normalised confusion matrices, MLP vs CNN
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"wspace": 0.35})
for a, e, name in ((ax[0], e1, "MLP"), (ax[1], e2, "CNN")):
    cm = np.array(e["confusion_matrix_test"], float); cm = 100*cm/cm.sum(1, keepdims=True)
    im = a.imshow(cm, cmap="Blues", norm=Normalize(0, 90))
    for i in range(10):
        for j in range(10):
            a.text(j, i, f"{cm[i,j]:.0f}", ha="center", va="center", fontsize=7,
                   color="white" if cm[i, j] > 50 else "black")
    a.set_xticks(range(10)); a.set_yticks(range(10))
    a.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8); a.set_yticklabels(CLASSES, fontsize=8)
    a.set(title=f"{name} (test accuracy {100*e['final']['test_acc']:.1f}%)", xlabel="Predicted class", ylabel="True class" if name == "MLP" else ""); a.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="% of true class")
fig.savefig(f"{OUT}/fig_ex1_ex2_confusion_matrices.png"); plt.close(fig)

# C. Ex3 train vs test with the gap, 30 and 60 epochs
r3, x3 = J("ex3_regularisation.json"), J("ex3_extension_60_epochs.json")
methods = ["Baseline", "Dropout (p=0.5)", "Augmentation", "Combined"]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
for a, S, t in ((ax[0], r3["summary"], "30 epochs (main protocol)"), (ax[1], x3["summary"], "60 epochs (extension)")):
    x = np.arange(4)
    tr = [100*S[m]["train_acc"]["mean"] for m in methods]; te = [100*S[m]["test_acc"]["mean"] for m in methods]
    trs = [100*S[m]["train_acc"]["std"] for m in methods]; tes = [100*S[m]["test_acc"]["std"] for m in methods]
    a.bar(x-0.19, tr, 0.38, yerr=trs, capsize=3, color="#8bb8ee", label="Training (clean)")
    a.bar(x+0.19, te, 0.38, yerr=tes, capsize=3, color="#1b4f93", label="Test")
    for i in range(4):
        a.text(x[i], max(tr[i], te[i])+2.2, f"gap {tr[i]-te[i]:.1f}", ha="center", fontsize=8.5, color=GREY)
        a.text(x[i]+0.19, te[i]-5, f"{te[i]:.1f}", ha="center", fontsize=8, color="white")
    a.set_xticks(x); a.set_xticklabels(methods, fontsize=9); a.set_ylim(50, 95); a.set_title(t); a.grid(axis="x", visible=False)
ax[0].set_ylabel("Accuracy (%)"); ax[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Exercise 3: training vs test accuracy and their difference (mean ± std, 3 seeds)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/fig_ex3_train_test_gap.png"); plt.close(fig)

# D. Ex3 mean training/validation accuracy curves per method (training dynamics)
names = {"Baseline": "Baseline", "Dropout p=0.5": "Dropout (p=0.5)",
         "Augmentation (all three)": "Augmentation", "Combined (p=0.5 + augmentation)": "Combined"}
fig, ax = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
for a, (run_name, label) in zip(ax, names.items()):
    runs = [r for r in r3["runs"] if r["config"]["name"] == run_name]
    for key, ls, lab in (("train_acc", "--", "training (running)"), ("val_acc", "-", "validation")):
        m = 100*np.mean([r["history"][key] for r in runs], 0)
        a.plot(ep, m, ls, c=BLUE if key == "val_acc" else ORANGE, label=lab)
    a.set(title=label, xlabel="Epoch")
ax[0].set_ylabel("Accuracy (%)"); ax[0].legend(fontsize=8)
fig.suptitle("Exercise 3: accuracy curves (mean of 3 seeds); training accuracy is the running value on the (augmented, dropout-active) training batches", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/fig_ex3_accuracy_curves.png"); plt.close(fig)

# E. Ex4 summary: validation accuracy, training loss, parameters vs width and depth (brief Ex4 req. 5-6)
r4 = J("ex4_width_depth.json"); S, C = r4["summary"], r4["complexity"]
sweeps = (("Width (depth 2)", ["16-32", "32-64", "64-128", "128-256"]),
          ("Depth (doubling channels)", ["32", "32-64", "32-64-128", "32-64-128-256"]))
fig, ax = plt.subplots(2, 3, figsize=(12.5, 6.3))
for row, (title, cfgs) in enumerate(sweeps):
    x = np.arange(len(cfgs)); col = BLUE if row == 0 else ORANGE
    va = [100*S[c]["val_acc"]["mean"] for c in cfgs]; vs = [100*S[c]["val_acc"]["std"] for c in cfgs]
    tl = [S[c]["train_loss"]["mean"] for c in cfgs]; ts = [S[c]["train_loss"]["std"] for c in cfgs]
    pa = [C[c]["parameters"]/1e3 for c in cfgs]; ma = [C[c]["macs"]/1e6 for c in cfgs]
    ax[row,0].errorbar(x, va, vs, marker="o", c=col, capsize=3)
    ax[row,1].errorbar(x, tl, ts, marker="o", c=col, capsize=3)
    ax[row,2].bar(x, pa, color=col, alpha=0.85)
    for i in range(len(cfgs)):
        ax[row,0].annotate(f"{va[i]:.1f}", (x[i], va[i]), textcoords="offset points", xytext=(8, -3), fontsize=8)
        ax[row,2].text(x[i], pa[i]*1.15, f"{pa[i]:.0f}k\n{ma[i]:.1f}M MACs", ha="center", fontsize=7.5)
    ax[row,2].set_yscale("log"); ax[row,2].set_ylim(10, 2500)
    bi = cfgs.index("32-64")
    for k in range(2): ax[row,k].scatter([bi], [ (va if k==0 else tl)[bi] ], s=160, facecolors="none", edgecolors="black", zorder=5)
    for k, yl in enumerate(("Validation accuracy (%)", "Training loss (clean, selected model)", "Parameters (thousands, log)")):
        ax[row,k].set_xticks(x); ax[row,k].set_xticklabels(cfgs, fontsize=8); ax[row,k].set_ylabel(yl, fontsize=8.5)
        ax[row,k].set_title(title, fontsize=9.5)
    ax[row,2].grid(axis="x", visible=False)
for k in range(3): ax[1,k].set_xlabel("Channels per convolution block")
fig.suptitle("Exercise 4: validation accuracy, training loss and model size against width (top) and depth (bottom); circle = baseline 32-64 (mean ± std, 3 seeds)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"{OUT}/fig_ex4_width_depth_summary.png"); plt.close(fig)
print("done")
