# Report figures (PUSL3123 Assessment 1)

Figures numbered in the order they appear in the report. All are drawn from `results/*.json`
or saved by the notebooks; no number was typed by hand.

| File | Report figure | Source |
|---|---|---|
| fig01_ex1_mlp_training_validation_curves.png | Figure 1 | Ex1 notebook |
| fig02_ex2_mlp_vs_cnn_train_val_curves.png | Figure 2 (training AND validation loss, as Ex2 req. 4 asks) | make_report_figures.py |
| fig03_ex2_cnn_feature_maps.png | Figure 3 | Ex2 notebook |
| fig04_ex1_ex2_confusion_matrices.png | Figure 4 | make_report_figures.py |
| fig05_ex3_augmentation_examples.png | Figure 5 | Ex3 notebook |
| fig06_ex3_train_test_gap.png | Figure 6 (train vs test and difference, Ex3 req. 3-4) | make_report_figures.py |
| fig07_ex3_validation_loss.png | Figure 7 | Ex3 notebook |
| fig08_ex3_ext_60_epoch_validation_loss.png | Figure 8 | Ex3 notebook (extension) |
| fig09_ex4_width_depth_summary.png | Figure 9 (val accuracy, training loss, parameters, Ex4 req. 5-6) | make_report_figures.py |
| fig10_ex4_accuracy_vs_parameters.png | Figure 10 | Ex4 notebook |
| fig11_ex4_ext_learning_rate.png | Figure 11 | Ex4 notebook (extension) |

`extra/` holds further figures that are not in the report (dataset samples, class distribution,
first-layer filters, per-class bars, ablation bars, dropout-rate selection, complexity bars,
overfitting gap, validation-loss curves, training-loss plot, per-method accuracy curves).
They can replace a report figure or go in an appendix, but each added caption counts toward the word limit.

Regenerate the new figures from the project folder:
`python report_figures/make_report_figures.py results report_figures_regenerated`
