"""Training, evaluation, and benchmarking module for SSVEP classifiers.

Compares CCA, FBCCA, SVM, and TRCA on real Wang 2016 preprocessed EEG trials.
Generates performance metrics, confusion matrices, and persists models.
"""

from pathlib import Path
import time
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.utils.config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    NUM_TARGETS,
)
from src.utils.logging import setup_logger
from src.dataset.loader import get_target_frequency_mapping, get_target_phase_mapping
from src.models.cca import CCAClassifier
from src.models.fbcca import FBCCAClassifier
from src.models.svm import SSVEPSVMClassifier
from src.models.trca import TRCAClassifier
from src.evaluation.metrics import compute_classification_metrics

logger = setup_logger("models_benchmark")


def load_preprocessed_subject(subject_id: str, duration_ms: int = 2000, roi: str = "occipital") -> dict:
    """Loads preprocessed NPZ dataset for a subject."""
    npz_path = PROCESSED_DATA_DIR / f"{subject_id}_roi-{roi}_dur-{duration_ms}ms.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {npz_path}")
    data = np.load(str(npz_path))
    return {
        "X": data["X"],
        "y": data["y"],
        "frequencies": data["frequencies"],
        "blocks": data["blocks"],
        "ch_names": data["ch_names"],
        "sfreq": float(data["sfreq"]),
        "duration": float(data["duration"]),
    }


def evaluate_model_on_subject(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selection_time_sec: float = 2.5,
) -> dict:
    """Fits and evaluates a model, measuring prediction latency and metrics."""
    # Fit model if supervised
    t_train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t_train_start

    # Predict with latency measurement
    t_inf_start = time.perf_counter()
    y_pred, confidences, margins = model.predict_with_confidence(X_test)
    inf_time_total = time.perf_counter() - t_inf_start
    latency_per_trial_ms = (inf_time_total / max(1, len(X_test))) * 1000.0

    # Compute metrics
    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        confidences=confidences,
        selection_time_sec=selection_time_sec,
        n_classes=NUM_TARGETS,
        latency_ms=latency_per_trial_ms,
    )
    metrics["train_time_sec"] = round(train_time, 4)
    metrics["total_inference_time_sec"] = round(inf_time_total, 4)
    metrics["mean_confidence"] = round(float(np.mean(confidences)), 4)
    metrics["mean_margin"] = round(float(np.mean(margins)), 4)

    return metrics, y_pred, confidences


def run_baseline_benchmark(
    subject_id: str = "sub-1",
    save_models: bool = True,
    generate_figures: bool = True,
) -> dict:
    """Runs comparative benchmark of CCA, FBCCA, SVM, and TRCA on a subject."""
    logger.info(f"=== Running Baseline Model Benchmark on {subject_id} ===")
    data = load_preprocessed_subject(subject_id)
    X = data["X"]
    y = data["y"]
    blocks = data["blocks"]
    sfreq = data["sfreq"]
    duration = data["duration"]
    selection_time = duration + 0.5  # Window duration + 0.5s cue

    freq_map = get_target_frequency_mapping()
    phase_map = get_target_phase_mapping()

    # Split into 5 training blocks (blocks 0..4, 200 trials) and 1 test block (block 5, 40 trials)
    train_mask = (blocks < 5)
    test_mask = (blocks == 5)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(f"Train set: {len(X_train)} trials across blocks 0..4; Test set: {len(X_test)} trials from block 5.")

    models = {
        "CCA": CCAClassifier(target_frequencies=freq_map, sfreq=sfreq, n_harmonics=5, target_phases=phase_map),
        "FBCCA": FBCCAClassifier(target_frequencies=freq_map, sfreq=sfreq, n_harmonics=5, n_subbands=5, target_phases=phase_map),
        "TRCA": TRCAClassifier(target_frequencies=freq_map, sfreq=sfreq, ensemble=True),
        "SVM": SSVEPSVMClassifier(target_frequencies=freq_map, sfreq=sfreq, n_harmonics=3, C=1.0),
    }

    results = {}
    predictions_dict = {}

    models_dir = RESULTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        metrics, y_pred, confidences = evaluate_model_on_subject(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            selection_time_sec=selection_time,
        )
        results[name] = metrics
        predictions_dict[name] = y_pred

        logger.info(
            f"{name} -> Accuracy: {metrics['accuracy']*100:.2f}%, "
            f"Macro-F1: {metrics['macro_f1']:.4f}, "
            f"ITR: {metrics['itr_bits_per_min']:.2f} bits/min, "
            f"Latency: {metrics['latency_ms']:.1f} ms/trial"
        )

        if save_models:
            model_file = models_dir / f"{subject_id}_{name.lower()}_model.joblib"
            model.save(model_file)
            logger.info(f"Saved {name} model to {model_file}")

    # Generate Confusion Matrix Figure for best model
    if generate_figures:
        best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
        generate_confusion_matrix_plot(
            y_true=y_test,
            y_pred=predictions_dict[best_model_name],
            model_name=best_model_name,
            subject_id=subject_id,
        )
        generate_comparison_bar_chart(results, subject_id=subject_id)

    # Save Results JSON
    results_path = REPORTS_DIR / f"{subject_id}_baseline_benchmark.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved baseline benchmark results to {results_path}")

    return results


def generate_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, subject_id: str):
    """Plots and saves confusion matrix heatmap."""
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import confusion_matrix
    classes = sorted(list(set(y_true).union(set(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name} on {subject_id} (Test Block 5)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Target ID", fontsize=11)
    ax.set_ylabel("True Target ID", fontsize=11)

    plt.tight_layout()
    fig_path = figures_dir / f"{subject_id}_{model_name.lower()}_confusion_matrix.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved confusion matrix figure to {fig_path}")


def generate_comparison_bar_chart(results: dict, subject_id: str):
    """Plots and saves comparison bar chart for Accuracy and ITR."""
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    models = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models]
    itrs = [results[m]["itr_bits_per_min"] for m in models]
    latencies = [results[m]["latency_ms"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Accuracy
    axes[0].bar(models, accuracies, color=["#3498db", "#2ecc71", "#9b59b6", "#e67e22"], width=0.55)
    axes[0].set_title(f"Decoding Accuracy ({subject_id})", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # ITR
    axes[1].bar(models, itrs, color=["#3498db", "#2ecc71", "#9b59b6", "#e67e22"], width=0.55)
    axes[1].set_title("Information Transfer Rate (ITR)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("ITR (bits / min)")
    for i, v in enumerate(itrs):
        axes[1].text(i, v + 1, f"{v:.1f}", ha="center", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    # Latency
    axes[2].bar(models, latencies, color=["#3498db", "#2ecc71", "#9b59b6", "#e67e22"], width=0.55)
    axes[2].set_title("Inference Latency per Trial", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Latency (ms)")
    for i, v in enumerate(latencies):
        axes[2].text(i, v + 0.5, f"{v:.1f}ms", ha="center", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = figures_dir / f"{subject_id}_model_comparison_barchart.png"
    plt.savefig(chart_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved comparison bar chart to {chart_path}")


if __name__ == "__main__":
    benchmark_results = run_baseline_benchmark("sub-1")
    print("=" * 60)
    print("BASELINE CLASSIFIER BENCHMARK COMPLETED")
    for m, r in benchmark_results.items():
        print(f"{m:6s} | Acc: {r['accuracy']*100:5.1f}% | F1: {r['macro_f1']:6.4f} | ITR: {r['itr_bits_per_min']:6.1f} bpm | Lat: {r['latency_ms']:5.1f} ms")
    print("=" * 60)
