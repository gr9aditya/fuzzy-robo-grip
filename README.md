# Fuzzy Robo Grip

Fuzzy Robo Grip is a research and teaching prototype exploring fuzzy logic in
robotic grasping. It contains two related but distinct parts: an interactive
grip-force estimator and an evaluation of tactile slip-detection models.

## Project components

- **[GripLab](./app/README.md)** is a local Python web app that estimates the
  total normal force for a stationary, symmetric two-finger grasp. It combines
  a basic physics calculation with a transparent fuzzy interpolation of
  user-supplied mass and friction uncertainty.
- **[Tactile slip evaluation](./evaluation/README.md)** compares a decision
  tree with a Mamdani fuzzy classifier on real uSkin tactile data. It reports
  held-out experiment metrics and includes scripts for plots and an optional
  human evaluation survey.

These components address different tasks: the tactile dataset evaluates slip
classification; it does not validate GripLab's grip-force estimates or predict
the force needed to hold a glass.

## Quick start

### Run GripLab

Python 3.9 or later is recommended. No third-party packages are needed:

```bash
cd app
python server.py
```

GripLab opens locally at `http://127.0.0.1:8876`. See the [app guide](./app/README.md)
for options and details about the assumptions and calculation.

### Run the tactile evaluation

The evaluation uses Python 3.12 and the packages listed in
[`evaluation/requirements.txt`](./evaluation/requirements.txt). From the
repository root:

```bash
cd evaluation
python -m pip install -r requirements.txt
python train_decision_tree.py
```

The first run downloads the default dataset. To choose another supported
dataset, pass `--dataset brush`, `--dataset screwdriver`, or `--dataset all`.
See the [evaluation guide](./evaluation/README.md) for plotting results and
running the optional survey.

## Scope and limitations

GripLab is a software estimate for a simplified static grasp, not a robot
controller or a guarantee of safe handling. Its example object and contact
values are illustrative assumptions, and it has no hardware validation.
Likewise, slip-detection results depend on the dataset and sensor setup and
should not be assumed to transfer to other grippers or tasks.
