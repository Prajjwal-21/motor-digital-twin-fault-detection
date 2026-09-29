"""Train and compare a Transformer and an LSTM on the MATLAB digital-twin dataset.

Usage:
    python train.py

Outputs:
    checkpoints/transformer.pt, checkpoints/lstm.pt   model weights
    checkpoints/scaler.json                            train-set mean/std per feature
    checkpoints/split.json                             run_ids of train/val/test
    results/results.json                               all metrics
    results/comparison.md                              markdown comparison table
    results/confusion_<model>.png, results/loss_<model>.png
"""

import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMClassifier, TransformerClassifier

# ----------------------------- configuration -----------------------------
SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "motor_dataset.csv"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

FEATURES = ["current", "speed", "torque", "temperature"]
CLASS_NAMES = ["Healthy", "High Load", "Increased Friction"]
SEQ_LEN = 100

BATCH_SIZE = 32
LR = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5  # early stopping: stop after 5 epochs without val-loss improvement

MODEL_BUILDERS = {
    "Transformer": TransformerClassifier,
    "LSTM": LSTMClassifier,
}

# Colors (colorblind-safe palette, same as the MATLAB figure and dashboard)
BLUE, ORANGE = "#2a78d6", "#eb6834"


def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------- data ------------------------------------
def load_runs(csv_path: Path = DATA_CSV):
    """Load the long-format CSV and reshape to X (N, 100, 4), y (N,), run_ids (N,)."""
    df = pd.read_csv(csv_path).sort_values(["run_id", "time"])
    run_ids = df["run_id"].unique()
    X = df[FEATURES].to_numpy(dtype=np.float32).reshape(len(run_ids), SEQ_LEN, len(FEATURES))
    y = df.groupby("run_id")["label"].first().loc[run_ids].to_numpy(dtype=np.int64)
    return X, y, run_ids


def split_by_run(run_ids, y):
    """Stratified 70/15/15 split. Each run (a whole 10 s time series) lands in
    exactly one split, so no time steps of a test run are ever seen in training."""
    train_ids, temp_ids, _, y_temp = train_test_split(
        run_ids, y, test_size=0.30, stratify=y, random_state=SEED)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, stratify=y_temp, random_state=SEED)
    return np.sort(train_ids), np.sort(val_ids), np.sort(test_ids)


def standardize(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


# ------------------------------ training ---------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
    return total_loss / n


def train_model(name, X_train, y_train, X_val, y_val):
    """Train with Adam + cross-entropy and early stopping on validation loss."""
    set_seed()  # same initialization / batch order for every model
    model = MODEL_BUILDERS[name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=256)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"  [{name}] epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  [{name}] early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)  # keep the weights with the lowest val loss
    return model, history, best_epoch, train_time


@torch.no_grad()
def predict(model, X):
    """Return (predicted class, softmax probabilities) for a batch of runs."""
    model.eval()
    probs = torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()
    return probs.argmax(axis=1), probs


# ------------------------------- plots -----------------------------------
def plot_confusion(cm, name, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name}: test confusion matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_losses(history, best_epoch, name, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], color=BLUE, lw=2, label="Train")
    ax.plot(epochs, history["val_loss"], color=ORANGE, lw=2, label="Validation")
    ax.axvline(best_epoch, color="#52514e", ls="--", lw=1, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{name}: loss curves")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------- main -----------------------------------
def main():
    set_seed()
    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    X, y, run_ids = load_runs()
    print(f"Loaded {len(y)} runs, X shape {X.shape}, class counts {np.bincount(y).tolist()}")

    train_ids, val_ids, test_ids = split_by_run(run_ids, y)
    idx = {rid: i for i, rid in enumerate(run_ids)}
    tr, va, te = ([idx[r] for r in ids] for ids in (train_ids, val_ids, test_ids))
    print(f"Split (runs): train {len(tr)}, val {len(va)}, test {len(te)}")

    # Standardize with TRAIN statistics only (per feature, over all train time steps).
    mean = X[tr].reshape(-1, len(FEATURES)).mean(axis=0)
    std = X[tr].reshape(-1, len(FEATURES)).std(axis=0)
    X_train, X_val, X_test = (standardize(X[s], mean, std) for s in (tr, va, te))
    y_train, y_val, y_test = y[tr], y[va], y[te]

    (CKPT_DIR / "scaler.json").write_text(json.dumps(
        {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist()}, indent=2))
    (CKPT_DIR / "split.json").write_text(json.dumps(
        {"train": train_ids.tolist(), "val": val_ids.tolist(), "test": test_ids.tolist()}))

    results = {"dataset": {"n_runs": int(len(y)), "n_train": len(tr), "n_val": len(va),
                           "n_test": len(te), "seq_len": SEQ_LEN, "features": FEATURES},
               "models": {}}

    for name in MODEL_BUILDERS:
        print(f"\nTraining {name}...")
        model, history, best_epoch, train_time = train_model(name, X_train, y_train, X_val, y_val)
        y_pred, _ = predict(model, X_test)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "confusion_matrix": cm.tolist(),
            "n_params": count_params(model),
            "train_time_s": round(train_time, 1),
            "epochs_run": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "history": history,
        }
        results["models"][name] = metrics
        print(f"  [{name}] test accuracy {metrics['accuracy']:.4f}, macro F1 {metrics['macro_f1']:.4f}")

        key = name.lower()
        torch.save(model.state_dict(), CKPT_DIR / f"{key}.pt")
        plot_confusion(cm, name, RESULTS_DIR / f"confusion_{key}.png")
        plot_losses(history, best_epoch, name, RESULTS_DIR / f"loss_{key}.png")

    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))

    rows = ["| Model | Test accuracy | Macro F1 | Parameters | Training time (s) | Epochs (best) |",
            "|---|---|---|---|---|---|"]
    for name, m in results["models"].items():
        rows.append(f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['n_params']:,} "
                    f"| {m['train_time_s']:.1f} | {m['epochs_run']} ({m['best_epoch']}) |")
    table = "\n".join(rows)
    (RESULTS_DIR / "comparison.md").write_text(
        f"Test set: {len(te)} runs (stratified, split by run_id). CPU training.\n\n{table}\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
