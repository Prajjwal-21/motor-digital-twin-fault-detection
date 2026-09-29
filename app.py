"""Streamlit dashboard: Motor Digital Twin — AI Fault Detection.

Usage:
    streamlit run app.py

Shows held-out TEST runs from the MATLAB digital twin and the live predictions
of the trained Transformer and LSTM (run train.py first).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

from train import CLASS_NAMES, FEATURES, MODEL_BUILDERS, SEQ_LEN, standardize

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# One fixed color per class (colorblind-safe; same as the MATLAB figure)
CLASS_COLORS = {"Healthy": "#2a78d6", "High Load": "#eb6834", "Increased Friction": "#1baf7a"}
SIGNAL_LABELS = {"current": "Current [A]", "speed": "Speed [RPM]",
                 "torque": "Torque [N·m]", "temperature": "Temperature [°C]"}

st.set_page_config(page_title="Motor Digital Twin — AI Fault Detection", page_icon="⚙️", layout="wide")


# ----------------------------- cached loading -----------------------------
@st.cache_resource
def load_models():
    """Build both networks and load their trained weights (cached across reruns)."""
    models = {}
    for name, builder in MODEL_BUILDERS.items():
        model = builder()
        model.load_state_dict(torch.load(CKPT_DIR / f"{name.lower()}.pt", map_location="cpu"))
        model.eval()
        models[name] = model
    return models


@st.cache_resource
def load_test_data():
    """Load the MATLAB CSV, keep only the test runs, and attach run metadata."""
    split = json.loads((CKPT_DIR / "split.json").read_text())
    scaler = json.loads((CKPT_DIR / "scaler.json").read_text())
    df = pd.read_csv(ROOT / "data" / "motor_dataset.csv")
    df = df[df["run_id"].isin(split["test"])].sort_values(["run_id", "time"])

    meta_path = ROOT / "data" / "run_metadata.csv"
    meta = pd.read_csv(meta_path).set_index("run_id") if meta_path.exists() else None
    results = json.loads((RESULTS_DIR / "results.json").read_text())
    return df, np.array(scaler["mean"]), np.array(scaler["std"]), meta, results


@torch.no_grad()
def predict_run(models, run_df, mean, std):
    """Return {model_name: softmax probabilities (3,)} for one run."""
    x = run_df[FEATURES].to_numpy(dtype=np.float32).reshape(1, SEQ_LEN, len(FEATURES))
    x = torch.from_numpy(standardize(x, mean, std))
    return {name: torch.softmax(m(x), dim=1)[0].numpy() for name, m in models.items()}


# ------------------------------- charts -----------------------------------
def signal_figure(run_df, color, fault_onset=None):
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(SIGNAL_LABELS.values()),
                        horizontal_spacing=0.08, vertical_spacing=0.14)
    for k, feature in enumerate(FEATURES):
        row, col = k // 2 + 1, k % 2 + 1
        fig.add_trace(go.Scatter(x=run_df["time"], y=run_df[feature], mode="lines",
                                 line=dict(color=color, width=2), name=SIGNAL_LABELS[feature],
                                 hovertemplate="t = %{x:.1f} s<br>%{y:.3f}<extra></extra>"),
                      row=row, col=col)
        if fault_onset is not None:
            fig.add_vline(x=fault_onset, line_dash="dash", line_color="#8a8983", row=row, col=col)
        if row == 2:  # x-axis title on the bottom row only, so it can't collide with titles
            fig.update_xaxes(title_text="Time [s]", row=row, col=col)
    if fault_onset is not None:
        fig.add_annotation(text=f"dashed line = fault onset ({fault_onset:.1f} s)",
                           xref="paper", yref="paper", x=1, y=1.12, showarrow=False,
                           font=dict(size=12))
    fig.update_layout(height=560, showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def confidence_figure(probs):
    fig = go.Figure(go.Bar(
        x=probs * 100, y=CLASS_NAMES, orientation="h",
        marker=dict(color=[CLASS_COLORS[c] for c in CLASS_NAMES], cornerradius=4),
        text=[f"{p:.1%}" for p in probs], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(height=190, margin=dict(l=10, r=50, t=10, b=10),
                      xaxis=dict(range=[0, 100], title="Confidence [%]"),
                      yaxis=dict(autorange="reversed"))
    return fig


# --------------------------------- app ------------------------------------
st.title("Motor Digital Twin — AI Fault Detection")
st.caption("Signals come from a MATLAB DC-motor digital twin (ode45). "
           "Only held-out **test** runs are shown — the models never saw them during training.")

models = load_models()
test_df, mean, std, meta, results = load_test_data()
labels_per_run = test_df.groupby("run_id")["label"].first()

# ---- Sidebar controls ----
st.sidebar.header("Controls")
condition = st.sidebar.selectbox("Motor condition", CLASS_NAMES)
random_clicked = st.sidebar.button("🎲 Random Run", width="stretch")
model_choice = st.sidebar.radio("Model", list(MODEL_BUILDERS), horizontal=True)

candidates = labels_per_run[labels_per_run == CLASS_NAMES.index(condition)].index.to_numpy()
# Pick a new run when the condition changes or the button is pressed; otherwise keep it.
if random_clicked or st.session_state.get("condition") != condition:
    st.session_state.condition = condition
    st.session_state.run_id = int(np.random.choice(candidates))
run_id = st.session_state.run_id
run_df = test_df[test_df["run_id"] == run_id]
st.sidebar.caption(f"Showing test run **#{run_id}** ({len(candidates)} test runs in this class)")

true_label = CLASS_NAMES[int(run_df["label"].iloc[0])]
fault_onset = None
if meta is not None and true_label != "Healthy":
    fault_onset = float(meta.loc[run_id, "fault_onset"])

# ---- Metric cards ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg current", f"{run_df['current'].mean():.2f} A")
c2.metric("Avg speed", f"{run_df['speed'].mean():,.0f} RPM")
c3.metric("Final temperature", f"{run_df['temperature'].iloc[-1]:.1f} °C")
c4.metric("True condition", true_label)

# ---- Signals + predictions ----
left, right = st.columns([2.2, 1])
with left:
    st.subheader("Sensor signals")
    st.plotly_chart(signal_figure(run_df, CLASS_COLORS[true_label], fault_onset),
                    width="stretch")

with right:
    st.subheader("AI diagnosis")
    probs = predict_run(models, run_df, mean, std)
    chosen = probs[model_choice]
    pred = CLASS_NAMES[int(chosen.argmax())]
    verdict = "✅ correct" if pred == true_label else "❌ wrong"
    st.markdown(f"**{model_choice}** predicts **{pred}** ({chosen.max():.1%}) — {verdict}")
    if meta is not None and true_label != "Healthy":
        st.caption(f"Ground truth: {true_label}, severity {meta.loc[run_id, 'severity']:.2f}x, "
                   f"onset {fault_onset:.1f} s")
    # Show both models, the selected one first
    for name in sorted(probs, key=lambda n: n != model_choice):
        p = probs[name]
        mark = "✅" if CLASS_NAMES[int(p.argmax())] == true_label else "❌"
        st.markdown(f"**{name}** {mark}")
        st.plotly_chart(confidence_figure(p), width="stretch", key=f"conf_{name}")

# ---- Results table ----
st.divider()
st.subheader("Model comparison on the test set")
table = pd.DataFrame([
    {"Model": name, "Test accuracy": f"{m['accuracy']:.3f}", "Macro F1": f"{m['macro_f1']:.3f}",
     "Parameters": f"{m['n_params']:,}", "Training time (s)": f"{m['train_time_s']:.1f}",
     "Epochs (best)": f"{m['epochs_run']} ({m['best_epoch']})"}
    for name, m in results["models"].items()])
st.dataframe(table, hide_index=True, width="stretch")
st.caption(f"Test set: {results['dataset']['n_test']} runs, split by run_id (stratified 70/15/15).")

with st.expander("Confusion matrices"):
    cols = st.columns(len(results["models"]))
    for col, name in zip(cols, results["models"]):
        col.image(str(RESULTS_DIR / f"confusion_{name.lower()}.png"), caption=name)
