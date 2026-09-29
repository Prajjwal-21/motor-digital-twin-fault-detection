# Motor Digital Twin — AI Fault Detection

**▶ Live demo:** https://motor-digital-twin-fault-detection-9hka6gsfntwxv2xzma5jan.streamlit.app
(hosted on Streamlit Community Cloud; if it has been idle, click "Yes, get this app back up!" and wait ~30 s)

A physics-based **digital twin of a DC motor in MATLAB** generates labeled sensor data, and two
**PyTorch** sequence models (a **Transformer** and an **LSTM** baseline) learn to diagnose the motor's
condition from 10 seconds of current, speed, torque, and temperature. A **Streamlit** dashboard shows
held-out test runs with live predictions.

Three conditions are classified:

| Label | Condition | What changes in the twin |
|---|---|---|
| 0 | Healthy | nothing |
| 1 | High Load | load torque `T_L` is multiplied by a severity factor after the fault onset |
| 2 | Increased Friction | friction coefficient `B` is multiplied by a severity factor after the fault onset |

![Example signals from the MATLAB digital twin](results/matlab_example_signals.png)

## Architecture

```
 ┌────────────────────────────┐      ┌──────────────────────┐
 │   MATLAB Digital Twin      │      │  data/               │
 │  motor_params.m            │      │  motor_dataset.csv   │
 │  dc_motor_ode.m  (physics) │ ───► │  900 runs x 100 steps│
 │  simulate_motor.m (ode45)  │      │  run_metadata.csv    │
 │  generate_dataset.m        │      └──────────┬───────────┘
 └────────────────────────────┘                 │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │ train.py (PyTorch)           │
                                 │  split by run_id 70/15/15    │
                                 │  standardize (train stats)   │
                                 │   ┌─────────┐  ┌───────────┐ │
                                 │   │  LSTM   │  │Transformer│ │
                                 │   └─────────┘  └───────────┘ │
                                 └──────────────┬───────────────┘
                                                │ checkpoints/ + results/
                                                ▼
                                 ┌──────────────────────────────┐
                                 │ app.py (Streamlit dashboard) │
                                 │  signals, predictions,       │
                                 │  confidence, comparison      │
                                 └──────────────────────────────┘
```

## Project structure

```
matlab/        digital twin + dataset generation
  motor_params.m      parameters of a small 24 V DC motor (struct)
  dc_motor_ode.m      the three differential equations + fault injection
  simulate_motor.m    ode45 over 0-10 s, resampled to 100 steps, sensor noise -> table
  generate_dataset.m  900 randomized runs -> data/motor_dataset.csv
  plot_examples.m     one run per class -> results/matlab_example_signals.png
  main.m              runs generate_dataset + plot_examples
data/          CSVs written by MATLAB
models/        transformer.py, lstm.py
train.py       trains and compares both models
app.py         Streamlit dashboard
checkpoints/   trained weights, scaler, train/val/test split
results/       metrics, comparison table, plots
```

## The motor physics (plain English)

The twin has three state variables: current `i`, speed `w`, and winding temperature `Temp`.

```
L    di/dt = V - R*i - Ke*w                        electrical
J    dw/dt = Kt*i - B*w - T_L                      mechanical
C_th dT/dt = i^2*R + B*w^2 - (Temp - T_amb)/R_th   thermal
```

1. **Voltage → current.** The supply voltage `V` pushes current through the winding resistance `R`
   (and inductance `L`). As the motor spins it generates a **back-EMF** `Ke*w` that opposes the supply,
   so the faster it spins, the less current flows.
2. **Current → torque.** The electromagnetic torque is proportional to current: `torque = Kt*i`.
3. **Torque → speed.** The motor accelerates (inertia `J`) until its torque is balanced by
   **friction** `B*w` (grows with speed) plus the **load** `T_L` (constant).
4. **Load and friction.** Anything that asks for more torque makes the motor slow down a little. Less
   speed means less back-EMF, so more current flows, which produces the extra torque. A new steady
   state follows with **lower speed and higher current**.
5. **Heating.** Copper losses `i^2*R` and friction losses `B*w^2` heat the motor; heat leaks to the
   ambient air through thermal resistance `R_th`. The thermal constants are deliberately scaled down
   (time constant 6 s instead of minutes) so heating is visible within a 10 s simulation.

The nominal motor settles at about 3400 RPM and 1.3 A at 24 V.

### How each fault changes the signals

| Signal | High Load (`T_L` ×s) | Increased Friction (`B` ×s) |
|---|---|---|
| Current / torque | step **up** at the onset | step **up** at the onset |
| Speed | step **down** | step **down** |
| Temperature | rises a bit faster (more `i^2 R`) | rises **much** faster (more `i^2 R` **and** `B w^2`) |

Both faults look almost the same electrically. At steady state `V = R*i + Ke*w`, so for *any* extra
torque demand the current rise and speed drop are tied together the same way. The main physical
difference is **where the power goes**: extra load power leaves through the shaft, but extra friction
power is converted to **heat inside the motor**. The temperature curve is therefore the key signal for
separating the two faults (see the example figure: at 2× severity, friction ends about 20 °C hotter than
high load).

## How the dataset was made harder, and why

A dataset where every healthy motor is identical and every fault is strong is solved at 100%
by a threshold on the current, which proves nothing. To force the models to learn the *pattern*
of a fault rather than memorize absolute values, every one of the 900 runs is randomized
(`generate_dataset.m`, all knobs in one `cfg` struct):

| Knob | Range | Why it makes the task harder |
|---|---|---|
| Fault onset | 2–6 s | the model must find the change anywhere in the window |
| Fault severity | 1.3× (mild) – 3× (strong) | mild faults give small steps, close to healthy |
| Supply voltage | 20–28 V | changes speed, current, and heating of *healthy* runs |
| Nominal load | ±30% | a healthy heavily loaded motor looks like a mild High Load fault |
| Nominal friction | ±30% | a healthy stiff motor looks like a mild friction fault |
| Ambient temperature | 20–35 °C | absolute temperature alone isn't informative |
| Sensor noise | 1% Gaussian per channel | realistic measurement noise |

Because operating ranges overlap, **absolute levels overlap between classes**. The model has to
detect the *step change* at the onset and the *shape* of the temperature rise.

The plan was to make the data harder still (lower minimum severity, wider ranges) if both models
scored ~100%. That wasn't needed: the LSTM reaches 93%, so the dataset already separates the models,
and the results below come from this dataset without further tuning.

## Models

| | Transformer | LSTM |
|---|---|---|
| Input | (batch, 100 steps, 4 features) | same |
| Core | Linear 4→64, sinusoidal positional encoding, 2 encoder layers (4 heads, FF 128, dropout 0.1) | 2-layer LSTM, hidden 64, dropout 0.1 |
| Sequence summary | global average pooling over time | last hidden state of the top layer |
| Head | Linear 64→3 | Linear 64→3 |

Training: Adam (lr 1e-3), cross-entropy, batch 32, up to 30 epochs, early stopping on validation loss
(patience 5, best weights restored), fixed seeds. Runs on a laptop CPU in well under a minute per model.

## Results

Test set: 135 runs (45 per class), never used for training or model selection. MacBook (Apple M2) CPU.
Accuracy and F1 are fully reproducible (fixed seeds); training time is wall-clock and varies a few seconds between runs.

| Model | Test accuracy | Macro F1 | Parameters | Training time (s) | Epochs (best) |
|---|---|---|---|---|---|
| Transformer | 0.993 | 0.993 | 67,459 | 8.5 | 14 (9) |
| LSTM | 0.933 | 0.934 | 51,395 | 14.4 | 30 (26) |

| Transformer | LSTM |
|---|---|
| ![Transformer confusion matrix](results/confusion_transformer.png) | ![LSTM confusion matrix](results/confusion_lstm.png) |
| ![Transformer loss](results/loss_transformer.png) | ![LSTM loss](results/loss_lstm.png) |

**What the errors say**
- The Transformer makes **one** mistake: a *mild* (1.38×) friction fault predicted as High Load.
  That is exactly the case where the two faults are physically closest.
- The LSTM makes 9 mistakes; **7 are Increased Friction runs predicted as Healthy**, mostly with
  early onsets (≤ 4.2 s). To spot the fault, a model has to compare signal levels before and after
  the onset. Attention compares any two time steps directly, while the LSTM must carry the
  pre-fault level through its memory.
- The LSTM was still improving when it hit the 30-epoch cap (best epoch 26), so with more epochs the
  gap would likely shrink. The comparison is at an equal, small training budget.
- 135 test runs is a small test set: one run is 0.7 percentage points. Treat the numbers as indicative.

## Dashboard

![Streamlit dashboard](results/dashboard_screenshot.png)

Choose a condition and a model in the sidebar, and press **Random Run** to load another held-out
test run. The dashboard shows metric cards, the four signals (with the fault onset marked), the true
label, both models' predictions with confidence bars, and the comparison table.

## How to run

Tested on macOS (Apple Silicon) with MATLAB R2025a and Python 3.14.

```bash
# 0) Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Digital twin + dataset (about 1 min; or open matlab/main.m in MATLAB and press Run)
/Applications/MATLAB_R2025a.app/bin/matlab -batch "cd matlab; main"

# 2) Train and compare both models (about 1 min on CPU)
python train.py

# 3) Dashboard locally (http://localhost:8501) — or use the live demo link above
streamlit run app.py
```

The generated CSV, trained checkpoints, and results are included, so you can run step 3
directly without MATLAB.

## Limitations

- **Simulated data only.** The twin is an idealized linear DC motor model. There's no real hardware,
  no bearing wear, no brush arcing, no temperature-dependent resistance, and no sensor drift.
- **Scaled thermal constants.** Heating is sped up so it's visible in 10 s. Real motors heat over minutes.
- **Only three clean classes**, one fault at a time, each a step change. Real faults can develop
  gradually and combine.
- **Small test set** (135 runs) and a single seed, so there are no confidence intervals.
- A model trained on this twin **will not transfer to a real motor** without real data (sim-to-real gap).
