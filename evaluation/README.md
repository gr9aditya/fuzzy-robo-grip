# Tactile slip detection: decision tree vs. Mamdani fuzzy system

This project compares two ways to classify **slip vs. no slip** from the real
uSkin tactile data in the
[ARQ-CRISP slip detection dataset](https://github.com/ARQ-CRISP/slip_detection_dataset_2021).
The dataset contains tactile sensor signals and manually annotated grasp/slip
events for a spool of solder, a brush, and a screwdriver.

> **Scope:** this dataset does not contain glasses, water volumes, grip force,
> or motor power. It cannot train or validate a model for predicting the power
> needed to hold a glass as it is filled. That application needs measurements
> from the intended gripper and glass (at minimum water mass/volume, force or
> motor command, and slip/success labels). This implementation uses the linked
> dataset for the related task it actually supports: tactile slip detection.

## Run

The project uses Python 3.12. Install the dependencies and run the comparison:

```bash
python -m pip install -r requirements.txt
python train_decision_tree.py
```

The first run downloads the spool-of-solder HDF5 dataset (about 143 MB) from
GitHub into `data/`. The source files are Git LFS-backed; the script downloads
the actual HDF5 data rather than the small Git LFS pointer file. Select another
object or all three objects with:

```bash
python train_decision_tree.py --dataset brush
python train_decision_tree.py --dataset screwdriver
python train_decision_tree.py --dataset all
```

`--data-dir`, `--output-dir`, and `--seed` can also be supplied. Dataset files
and generated results are ignored by Git.

## What is compared

- **Inputs:** per-sample mean shear-to-normal-force ratio and relative change
  in shear readings, derived from the sensor's 18 pins and three axes. Samples
  with labels 2, 3, or 6 are excluded. Labels 1 and 5 mean slip; labels 0 and 4
  mean no slip.
- **Decision tree:** a shallow, class-balanced scikit-learn classifier.
- **Mamdani system:** low/high input membership sets, four rules that treat
  high shear change as elevated slip risk (especially with high shear relative
  to normal force), min implication/max aggregation, and centroid
  defuzzification. Feature scaling is learned from training experiments only.
- **Evaluation:** experiments (not individual adjacent samples) are separated
  into training, validation, and test groups. Both models select a
  validation-only threshold to maximize slip F1, then are evaluated on the same
  held-out test experiments.
- **Results:** `results/metrics.csv` contains accuracy, balanced accuracy,
  precision, recall, F1, PR-AUC, ROC-AUC, and the selected threshold.
  `results/test_predictions.csv` contains test labels and continuous scores for
  both models.

## Create evaluation figures

After running the model comparison, generate publication-quality PNG and
vector PDF charts from the saved test predictions and metrics:

```bash
python plot_evaluation.py
```

By default, the script saves precision–recall curves, an ROC curve, side-by-side
confusion matrices, and a metric comparison chart under `results/figures/`.
The curves are calculated from test scores; confusion matrices and metric
values use the validation-selected thresholds recorded with the results.

## Run the human evaluation survey

The separate Streamlit app presents 20 balanced, blind examples from the
same held-out test experiments. Participants classify each tactile example
as slip, no slip, or not sure, and rate their confidence. Labels and both
model predictions are only revealed after all answers are submitted.

```bash
python -m pip install -r requirements.txt
python train_decision_tree.py
streamlit run survey_app.py
```

Open the local URL printed by Streamlit. The app does not collect names or
contact information. Completed responses are saved as one anonymous CSV per
session under `results/survey_responses/`; the app shows the human/model
comparison only after the survey is finished. These files are local to the
machine running the app and should be handled as research data. The default
survey is local-only; hosting it publicly requires a hosting service, persistent
storage, and suitable participant/privacy information.

Each survey uses the same fixed set of 20 held-out samples (10 per class) with
their display order shuffled per participant. This balanced sample is useful
for comparing slip and no-slip judgments, but its accuracy is not an estimate
of performance at the dataset's natural slip frequency.

Because slip labels are relatively rare, compare recall, precision, F1, and
PR-AUC rather than relying on accuracy alone. The dataset was collected with a
specific tactile sensor and robot setup; results do not directly transfer to a
glass-handling gripper.
