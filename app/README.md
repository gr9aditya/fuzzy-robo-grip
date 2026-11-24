# GripLab — explainable grip-force estimator

A small Python application with a local browser interface. It starts with an object, asks you to review its assumptions, and explains the calculation behind an estimated grip force.

**No GPU, Isaac Sim, internet connection, or third-party Python packages are required.** This version estimates a stationary grasp. It does not control a robot or simulate motion.

## Run the app

On Windows, double-click **Launch GripLab.cmd**. Keep the server console open while using the app. The launcher uses installed Python, or this computer's bundled Codex Python if available.

Alternatively, open a terminal in this folder and run:

```text
python server.py
```

The browser opens at **http://127.0.0.1:8876**. Python 3.9 or later is recommended. No `pip install` is needed.

- Stop the app with **Ctrl+C** in the server console, or close that console.
- If port 8876 is already occupied, run `python server.py --port 8877`.
- To start the server without automatically opening a browser, use `python server.py --no-browser`.
- Open the server address, not `index.html` directly: calculations run in Python through the local server.
- The server binds only to the local computer. It does not publish the application online.

## Two-minute midterm demonstration

1. **Choose “Empty drinking glass.”** Explain that this selects an editable example profile, not a recognized real object.
2. **Review the assumptions:** mass **250 g**, friction coefficient **0.40**, possible mass increase **10%**, possible friction reduction **25%**. Leave the allowable force blank. The example values are assumptions, not measured specifications.
3. **Calculate.** The main answer is approximately **9.0 N total**, or **4.5 N per finger**. The unrounded values are **8.9925 N total** and **4.49625 N per finger**.
4. **Follow the calculation:** 0.250 kg × 9.81 = **2.4525 N weight**; divide by 0.40 = **6.13125 N nominal total grip**; multiply by the fuzzy allowance **1.466666…** = **8.9925 N**.
5. **Open the fuzzy explanation.** The 10% mass bound belongs 0.6 to Low and 0.4 to Medium. The 25% friction bound belongs fully to Medium. Two rules contribute: `0.6 × 1.333333… + 0.4 × 1.666666… = 1.466666…`.
6. **Compare “Filled drinking glass.”** Its example mass is **500 g**, with the same other assumptions. The answer doubles to **17.985 N total**. This demonstrates a meaningful relationship between mass and required normal force.
7. **Return to the 250 g glass and enter a 5 N allowable total force.** The estimate remains 8.9925 N. The app explains that 5 N is even below the **6.13125 N nominal threshold**. It does not hide the conflict by clipping its estimate.

Use **Save calculation** to download the input record, rules, intermediate values, and result as JSON. This is useful evidence for the presentation. It is a model calculation, not an experimental measurement.

Suggested explanation:

> “Our prototype translates object and contact assumptions into an initial grip-force estimate. Physics supplies the nominal force. Fuzzy logic blends uncertainty scenarios into an allowance. We show the inputs and every calculation so the recommendation can be questioned and checked.”

## How the calculation works

### 1. Nominal static holding requirement

The model assumes an object held vertically between two equal opposing contacts. Each contact applies half of the total inward normal force. With the same Coulomb friction coefficient at both contacts:

```text
weight = mass_kg × 9.81
nominal_total_normal_force = weight / friction_coefficient
force_per_finger = total_normal_force / 2
```

“Grip force” here means the **sum of the two normal contact forces**. It is not the object weight, motor torque, or a vendor-specific gripper setting. Vendor force conventions must be checked before comparing specifications.

### 2. User-supplied uncertainty bounds

Let `p` be the possible relative mass increase and `q` the possible relative friction reduction. Both range from 0 to 0.50. For example, 10% and 25% mean `p = 0.10` and `q = 0.25`.

The direct worst-case calculation within these assumptions is:

```text
direct_bound_multiplier = (1 + p) / (1 - q)
direct_bound_force = nominal_total_force × direct_bound_multiplier
```

These percentages are **assumed bounds**, not probabilities, measured confidence intervals, or automatic material classifications.

### 3. Zero-order Sugeno fuzzy interpolation

Each uncertainty input has Low, Medium and High membership functions at **0%, 25%, and 50%**. They overlap continuously and sum to one. The nine rule outputs come from the direct physical uncertainty ratio at those anchor combinations:

| Mass bound / friction reduction | Low: 0% | Medium: 25% | High: 50% |
|---|---:|---:|---:|
| Low: 0% | 1.0000× | 1.3333× | 2.0000× |
| Medium: 25% | 1.2500× | 1.6667× | 2.5000× |
| High: 50% | 1.5000× | 2.0000× | 3.0000× |

For each rule, activation is the product of the two membership degrees. The output multiplier is the weighted average of the nine rule consequents. The app multiplies the nominal force by this multiplier. Calculations use unrounded values; display values are rounded.

This is a transparent teaching use of fuzzy logic. It is **not evidence that fuzzy logic is necessary or more accurate** than the direct equation. Because the friction term `1 / (1 − q)` is convex, this piecewise-linear interpolation equals or exceeds the direct bound over the supported domain. It equals the bound at the friction anchors, including the default 25% case. The interface shows both outputs so that this comparison is visible.

### 4. Check an optional supplied force limit

The limit is independent of the estimator. It never changes or clips the recommendation.

- **Unknown limit:** holding-force estimate only; damage tolerance is unknown.
- **Below nominal threshold:** no force within the supplied limit satisfies the model's nominal static requirement.
- **Below fuzzy estimate:** the limit covers the nominal threshold but not the full fuzzy allowance.
- **Within supplied limit:** a numerical comparison only, not proof of a secure or damage-free real grasp.

## What the example data means

The four example profiles use masses of 250 g, 500 g, 180 g and 750 g, and each starts with friction coefficient 0.40. These values are deliberately labelled **illustrative assumptions**. They are not product specifications or measured glass, phone or box data. Actual friction depends on both contact surfaces and their condition.

An object name is only a label and a way to select a starting profile. Changing the name alone does not change the calculation. Custom objects require supplied properties. Contact description documents an assumption; it does not trigger an undisclosed lookup.

## Scope and limitations

The model assumes a stationary vertical object, an established symmetric grasp, identical dry contacts, equal force sharing, and no torque or off-centre load. It does not calculate crushing, scratching, acceleration, rotation, slip recovery, or gripper actuator capability. Its uncertainty bound is conditional on the supplied bounds being meaningful for the actual contact.

The prototype provides no hardware validation, automatic material recognition, or guarantee of safe handling. Do not use these illustrative results as real robot settings.

## Why the interface is organized this way

- **Object first:** the user starts with a familiar task rather than unexplained fuzzy sliders.
- **Review before calculating:** assumptions are explicit and editable before the result appears.
- **One main answer:** total force and per-finger force are prominent and clearly distinguished.
- **Progressive detail:** the basic calculation is visible; rule tables, membership curves and scientific assumptions can be expanded when needed.
- **Traceability:** the input record and downloadable calculation connect the answer to its assumptions.
- **Error prevention:** validation rejects unsupported inputs, and force-limit conflicts remain visible.

These choices aim to reduce effort and make the estimate understandable. No usability study, business ROI, cost saving, or real-world performance improvement has been established.

## Verification and files

Run the computational checks from this folder:

```text
python -m unittest -v
```

The 12 tests cover the worked example, rule anchors, membership coverage, the analytical bound comparison, monotonicity, mass/friction scaling, total/per-finger convention, limit boundaries, invalid inputs, profile handling, and local HTTP integration. Supported mass range is 1–20,000 g; friction coefficient is 0.05–1.50; each uncertainty bound is 0–50%. These are software and mathematical checks, not physical experiments.

| File | Purpose |
|---|---|
| `model.py` | Validated physics and fuzzy calculations; example profiles |
| `server.py` | Local Python server and JSON endpoints |
| `index.html` | Three-step user interface |
| `app.js` | Form flow, result rendering, charts and JSON download |
| `style.css` | Responsive visual design |
| `test_model.py` | Computational tests |
| `test_server.py` | Local API integration tests |
| `Launch GripLab.cmd` | Windows launcher |

## Scientific references and disclosure

- [Modern Robotics: Coulomb friction](https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-2-1-friction/)
- [Robotiq 2F manual, section 5.2.1: payload and force](https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_General_PDF_20210623.pdf)

These sources support the friction relation and the need to distinguish force conventions. They do not validate this app's presets or uncertainty settings.

Codex assisted with implementation, tests, and documentation. The team should review the assumptions and explain the calculations in its own words during the midterm.
