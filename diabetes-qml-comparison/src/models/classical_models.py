"""Phase 5 (extended): train, tune, and compare three classical baselines
(Logistic Regression, Decision Tree, Random Forest) on the diabetes dataset,
and expose a standardized interface for the Phase 8 classical-vs-quantum
comparison.
"""

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Make `config`, `data_loader`, `preprocessing`, and sibling model modules
# importable regardless of how this file is loaded.
_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
_MODELS_DIR = Path(__file__).resolve().parent
for _path in (str(_PROJECT_ROOT), str(_SRC_DIR), str(_MODELS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from .quantum_qsvm_model import build_qsvm
    from .quantum_vqc_model import build_vqc
except ImportError:
    from quantum_qsvm_model import build_qsvm
    from quantum_vqc_model import build_vqc

import config
from data_loader import load_data
from preprocessing import preprocess


# ==========================================================================
# Model builders
# ==========================================================================

def build_logistic_regression() -> LogisticRegression:
    """Build an untrained Logistic Regression model."""
    return LogisticRegression(
        C=config.CLASSICAL_C,
        max_iter=config.CLASSICAL_MAX_ITER,
        solver=config.CLASSICAL_SOLVER,
        class_weight=config.CLASSICAL_CLASS_WEIGHT,
        random_state=config.CLASSICAL_RANDOM_STATE,
    )


def build_decision_tree() -> DecisionTreeClassifier:
    """Build an untrained Decision Tree model."""
    return DecisionTreeClassifier(
        max_depth=config.DT_MAX_DEPTH,
        min_samples_split=config.DT_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.DT_MIN_SAMPLES_LEAF,
        criterion=config.DT_CRITERION,
        random_state=config.DT_RANDOM_STATE,
    )


def build_random_forest() -> RandomForestClassifier:
    """Build an untrained Random Forest model."""
    return RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        max_features=config.RF_MAX_FEATURES,
        random_state=config.RF_RANDOM_STATE,
    )


MODEL_BUILDERS: Dict[str, Callable[[], Any]] = {
    "Logistic Regression": build_logistic_regression,
    "Decision Tree": build_decision_tree,
    "Random Forest": build_random_forest,
    "QSVM": build_qsvm,
    "VQC": build_vqc,
}

MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "description": "Linear model estimating class probabilities via a logistic function.",
        "type": "Linear",
        "key_parameters": ["C", "max_iter", "solver", "class_weight"],
    },
    "Decision Tree": {
        "description": "Single tree that splits on feature thresholds; non-linear, interpretable.",
        "type": "Tree-based",
        "key_parameters": ["max_depth", "min_samples_split", "min_samples_leaf", "criterion"],
    },
    "Random Forest": {
        "description": "Ensemble of decision trees combined by majority vote; robust and accurate.",
        "type": "Ensemble",
        "key_parameters": ["n_estimators", "max_depth", "min_samples_split", "max_features"],
    },
    "QSVM": {
        "description": "Quantum Support Vector Machine using a quantum kernel.",
        "type": "Quantum",
        "key_parameters": ["feature_map", "kernel", "C"],
    },
    "VQC": {
        "description": "Variational Quantum Classifier using a parameterized circuit.",
        "type": "Quantum",
        "key_parameters": ["feature_map", "ansatz", "optimizer"],
    },
}

# Direction in which "better" moves, per metric column.
_METRIC_DIRECTION = {
    "Accuracy": "max",
    "Precision": "max",
    "Recall": "max",
    "F1 Score": "max",
    "Train Accuracy": "max",
    "Training Time": "min",
    "Prediction Time": "min",
}


def get_all_models() -> Dict[str, Any]:
    """Build and return all three classical models, keyed by display name."""
    return {name: builder() for name, builder in MODEL_BUILDERS.items()}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Return the description, type, and key parameters for a model."""
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_INFO)}.")
    return MODEL_INFO[model_name]


# ==========================================================================
# Training & evaluation
# ==========================================================================

def train_all_models(
    models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train every model in `models`, timing each fit.

    Returns (trained_models, training_info) where training_info maps each
    model name to {'Training Time': secs, 'Train Accuracy': acc}.
    """
    trained_models: Dict[str, Any] = {}
    training_info: Dict[str, Dict[str, float]] = {}
    y_train = np.asarray(y_train).ravel()

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        train_pred = model.predict(X_train)
        trained_models[name] = model
        training_info[name] = {
            "Training Time": elapsed,
            "Train Accuracy": accuracy_score(y_train, train_pred),
        }

    return trained_models, training_info


def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_info: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate every trained model on the test set.

    Returns a dict keyed by model name with Accuracy, Precision, Recall,
    F1 Score, Prediction Time, and Confusion Matrix, merged with
    `training_info` (Training Time / Train Accuracy) when supplied.
    """
    results: Dict[str, Dict[str, Any]] = {}
    y_test = np.asarray(y_test).ravel()

    for name, model in trained_models.items():
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_time = time.perf_counter() - start

        entry = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "Prediction Time": pred_time,
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
        }
        if training_info and name in training_info:
            entry.update(training_info[name])

        results[name] = entry

    return results


def _results_to_frame(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Convert an evaluate_all_models() result dict into a display DataFrame."""
    columns = [
        "Accuracy", "Precision", "Recall", "F1 Score",
        "Training Time", "Train Accuracy", "Prediction Time",
    ]
    rows = {
        name: {col: entry.get(col, np.nan) for col in columns}
        for name, entry in results.items()
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Model"
    return df


def compare_all_models(
    results: Dict[str, Dict[str, Any]], plot: bool = False, save_path: Optional[Path] = None
) -> pd.DataFrame:
    """Build a comparison table from results and optionally save a bar chart."""
    df = _results_to_frame(results)
    if plot:
        save_path = save_path or (config.RESULTS_PATH / "classical_models_comparison.png")
        plot_model_comparison(df, save_path)
    return df


def _best_per_metric(df: pd.DataFrame) -> Dict[str, str]:
    """Return {metric: best_model_name} respecting each metric's direction."""
    best = {}
    for metric in df.columns:
        if metric not in _METRIC_DIRECTION or df[metric].isna().all():
            continue
        ascending = _METRIC_DIRECTION[metric] == "min"
        best[metric] = df[metric].sort_values(ascending=ascending).index[0]
    return best


def _fmt(col: str, val: float) -> str:
    if pd.isna(val):
        return "-"
    if col in ("Training Time", "Prediction Time"):
        return f"{val:.3f}s"
    return f"{val:.4f}"


def print_comparison_table(df: pd.DataFrame, title: str = "MODEL COMPARISON") -> None:
    """Pretty-print a comparison table, marking the best value per metric with a star."""
    best = _best_per_metric(df)
    col_width = max(17, max((len(col) for col in df.columns), default=0) + 2)
    total_width = 22 + col_width * len(df.columns)

    print(f"\n{title}")
    print("-" * total_width)
    print(f"{'Model':<22}" + "".join(f"{col:>{col_width}}" for col in df.columns))
    print("-" * total_width)

    for name, row in df.iterrows():
        cells = []
        for col in df.columns:
            text = _fmt(col, row[col])
            if best.get(col) == name:
                text = f"*{text}"
            cells.append(f"{text:>{col_width}}")
        print(f"{name:<22}" + "".join(cells))
    print("-" * total_width)

    for metric, name in best.items():
        print(f"* Best {metric}: {name} ({_fmt(metric, df.loc[name, metric])})")


# ==========================================================================
# Hyperparameter tuning
# ==========================================================================

TUNING_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["liblinear", "lbfgs"],
        "max_iter": [100, 500, 1000],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
}


def tune_all_models(
    X_train: np.ndarray, y_train: np.ndarray, cv: Optional[int] = None, verbose: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Run GridSearchCV for each model over its tuning grid.

    Returns (tuned_models, best_params) where tuned_models maps each name to
    its best fitted estimator, and best_params maps each name to
    {'best_params': ..., 'best_score': ...}.
    """
    cv = cv or config.CV_FOLDS
    tuned_models: Dict[str, Any] = {}
    best_params: Dict[str, Dict[str, Any]] = {}

    for name, builder in MODEL_BUILDERS.items():
        if name not in TUNING_GRIDS:
            if verbose:
                print(f"    Skipping {name} (no tuning grid defined)")
            continue
        if verbose:
            print(f"    Tuning {name}...")
        grid = GridSearchCV(
            builder(),
            TUNING_GRIDS[name],
            cv=cv,
            scoring=config.TUNING_SCORING,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        tuned_models[name] = grid.best_estimator_
        best_params[name] = {"best_params": grid.best_params_, "best_score": grid.best_score_}
        if verbose:
            print(f"      -> {grid.best_params_} (CV {config.TUNING_SCORING}={grid.best_score_:.4f})")

    return tuned_models, best_params


# ==========================================================================
# Model selection & reporting
# ==========================================================================

def analyze_model_selection(
    df: pd.DataFrame, overfit_threshold: float = 0.05, verbose: bool = True
) -> Dict[str, Any]:
    """Identify the best model per use case and flag any overfitting."""
    best = _best_per_metric(df)
    best_accuracy_model = best.get("Accuracy")
    fastest_model = best.get("Training Time")
    interpretable_candidates = [m for m in ("Decision Tree", "Logistic Regression") if m in df.index]

    overfit_flags = {}
    if "Train Accuracy" in df.columns:
        for name, row in df.iterrows():
            gap = row.get("Train Accuracy", np.nan) - row.get("Accuracy", np.nan)
            overfit_flags[name] = bool(gap > overfit_threshold)

    recommendations = {
        "best_overall": best_accuracy_model,
        "best_interpretability": interpretable_candidates[0] if interpretable_candidates else None,
        "best_accuracy": best_accuracy_model,
        "fastest_training": fastest_model,
        "overfitting_detected": any(overfit_flags.values()),
        "overfit_flags": overfit_flags,
    }

    if verbose:
        print("\nMODEL SELECTION TIPS")
        print("=" * 60)
        print(f"BEST OVERALL: {best_accuracy_model} (Accuracy: {df.loc[best_accuracy_model, 'Accuracy']:.4f})")
        print("\nRecommendations:")
        print(f"  - Use {best_accuracy_model} for best accuracy (quantum comparison baseline)")
        if recommendations["best_interpretability"]:
            print(f"  - Use {recommendations['best_interpretability']} for interpretability")
        if fastest_model:
            print(f"  - Use {fastest_model} for fastest training")
        print(f"  - {best_accuracy_model} suggested as baseline for Phase 8")
        if recommendations["overfitting_detected"]:
            flagged = [m for m, f in overfit_flags.items() if f]
            print(f"\n  Overfitting detected in: {', '.join(flagged)}")
        else:
            print("\n  No overfitting detected in any model")
        print("  Proceed to quantum models comparison")

    return recommendations


def plot_model_comparison(df: pd.DataFrame, save_path: Path) -> Path:
    """Save a grouped bar chart comparing Accuracy/Precision/Recall/F1 Score."""
    metrics = [m for m in ("Accuracy", "Precision", "Recall", "F1 Score") if m in df.columns]
    ax = df[metrics].plot(kind="bar", figsize=(10, 6))
    ax.set_title("Classical Models Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=0)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    return save_path


def _save_confusion_matrices(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, entry in results.items():
        cm = entry.get("Confusion Matrix")
        if cm is None:
            continue
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        fname = name.lower().replace(" ", "_") + "_confusion_matrix.png"
        plt.savefig(out_dir / fname)
        plt.close(fig)


def generate_comparison_report(
    df: pd.DataFrame,
    results: Dict[str, Dict[str, Any]],
    recommendations: Dict[str, Any],
    out_dir: Optional[Path] = None,
) -> Path:
    """Write a text report summarizing the comparison and save confusion matrices."""
    out_dir = out_dir or config.CLASSICAL_MODELS_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrices(results, config.CONFUSION_MATRICES_PATH)

    lines = [
        "=" * 60, "CLASSICAL MODELS COMPARISON REPORT", "=" * 60, "",
        "Summary table:", df.round(4).to_string(), "",
    ]
    if "Training Time" in df.columns:
        fastest = df["Training Time"].idxmin()
        lines.append(f"Fastest training: {fastest} ({df.loc[fastest, 'Training Time']:.3f}s)")
    lines += [
        "",
        f"Best overall (Phase 8 baseline): {recommendations.get('best_overall')}",
        f"Best interpretability: {recommendations.get('best_interpretability')}",
        f"Overfitting detected: {recommendations.get('overfitting_detected')}",
        "",
    ]

    report_path = out_dir / "comparison_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ==========================================================================
# Persistence
# ==========================================================================

def _model_filename(name: str) -> str:
    return name.lower().replace(" ", "_") + ".pkl"


def save_all_models(
    trained_models: Dict[str, Any], df: pd.DataFrame, models_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Persist each trained model plus the top-level comparison table/plot.

    Layout:
        results/classical_models_comparison.csv
        results/classical_models_comparison.png
        results/classical_models/<model>.pkl
    """
    models_dir = models_dir or config.CLASSICAL_MODELS_PATH
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in trained_models.items():
        joblib.dump(model, models_dir / _model_filename(name))

    csv_path = config.RESULTS_PATH / "classical_models_comparison.csv"
    plot_path = config.RESULTS_PATH / "classical_models_comparison.png"
    df.round(4).to_csv(csv_path)
    plot_model_comparison(df, plot_path)

    return {"models_dir": models_dir, "csv": csv_path, "plot": plot_path}


def load_all_models(models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load previously saved models from disk, keyed by display name."""
    models_dir = models_dir or config.CLASSICAL_MODELS_PATH
    loaded: Dict[str, Any] = {}
    for name in MODEL_BUILDERS:
        path = models_dir / _model_filename(name)
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded


# ==========================================================================
# Standardized interface for Phase 8
# ==========================================================================

def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Fit a single model in place and return it (standardized `train` hook)."""
    model.fit(X_train, y_train)
    return model


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Standardized `predict` hook."""
    return model.predict(X)


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Standardized `evaluate` hook returning the core classification metrics."""
    y_pred = predict(model, X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }


def get_model_interface(model_type: str) -> Dict[str, Any]:
    """Return a standardized {name, type, build, train, predict, evaluate}
    interface for `model_type`, so Phase 8 can drive classical and quantum
    models identically. `build` is a callable (not a pre-built instance) so
    a fresh, unfitted model can be constructed on demand.
    """
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{model_type}'. Choose from {list(MODEL_BUILDERS)}.")
    return {
        "name": model_type,
        "type": MODEL_INFO[model_type]["type"],
        "build": MODEL_BUILDERS[model_type],
        "train": train_model,
        "predict": predict,
        "evaluate": evaluate_model,
    }


def get_best_classical_model(
    df: Optional[pd.DataFrame] = None, trained_models: Optional[Dict[str, Any]] = None
) -> Tuple[str, Any, Dict[str, Any]]:
    """Return (name, fitted_model, metrics) for the best classical model by Accuracy.

    If `df`/`trained_models` are not supplied, runs the full pipeline first.
    """
    if df is None or trained_models is None:
        pipeline = run_complete_pipeline(verbose=False)
        df = pipeline["tuned_comparison"]
        trained_models = pipeline["tuned_models"]

    best_name = df["Accuracy"].idxmax()
    return best_name, trained_models[best_name], df.loc[best_name].to_dict()


def get_classical_results(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a comparison DataFrame into the standardized record list Phase 8 expects."""
    return [
        {
            "model_name": name,
            "model_type": "Classical",
            "accuracy": row.get("Accuracy"),
            "precision": row.get("Precision"),
            "recall": row.get("Recall"),
            "f1_score": row.get("F1 Score"),
            "training_time": row.get("Training Time"),
        }
        for name, row in df.iterrows()
    ]


# ==========================================================================
# Full pipeline
# ==========================================================================

def run_complete_pipeline(verbose: bool = True) -> Dict[str, Any]:
    """Run the complete classical-models workflow end to end: load data,
    build/train/evaluate the three models, tune them, compare results, save
    everything to disk, and return a dict with all intermediate artifacts
    for Phase 8 to consume programmatically.
    """
    pipeline_start = time.perf_counter()

    def log(msg: str = "") -> None:
        if verbose:
            print(msg)

    log("=" * 60)
    log("PHASE 5: CLASSICAL MODELS COMPARISON")
    log("=" * 60)

    log("\n[1] DATA PREPARATION")
    try:
        X, y = load_data()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not load diabetes dataset: {exc}") from exc
    df_full = X.copy()
    df_full["Outcome"] = y
    X_train, X_test, y_train, y_test = preprocess(df_full)
    log(f"    Loaded {len(df_full)} samples with {X_train.shape[1]} features")
    log(f"    Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    log("\n[2] BUILDING MODELS")
    models = get_all_models()
    for name in models:
        log(f"    Built {name}")

    log("\n[3] TRAINING MODELS")
    trained_models, training_info = train_all_models(models, X_train, y_train)
    for name, info in training_info.items():
        log(f"    {name:<22} {info['Training Time']:.3f}s   train acc={info['Train Accuracy']:.4f}")

    log("\n[4] EVALUATING MODELS - TEST SET")
    results = evaluate_all_models(trained_models, X_test, y_test, training_info)
    test_df = compare_all_models(results)
    if verbose:
        print_comparison_table(test_df, title="TEST SET RESULTS")

    log("\n[5] HYPERPARAMETER TUNING")
    tuned_models, best_params = tune_all_models(X_train, y_train, verbose=verbose)

    log("\n[6] TUNED MODEL COMPARISON")
    _, tuned_training_info = train_all_models(tuned_models, X_train, y_train)
    tuned_results = evaluate_all_models(tuned_models, X_test, y_test, tuned_training_info)
    tuned_df = compare_all_models(tuned_results)
    if verbose:
        print_comparison_table(tuned_df, title="TUNED RESULTS")
        improvement = tuned_df["Accuracy"] - test_df["Accuracy"]
        for name, delta in improvement.items():
            print(f"    {name:<22} improvement: {delta:+.4f}")

    log("\n[7] MODEL SELECTION")
    recommendations = analyze_model_selection(tuned_df, verbose=verbose)

    log("\n[8] RESULTS SAVED")
    paths = save_all_models(tuned_models, tuned_df)
    report_path = generate_comparison_report(tuned_df, tuned_results, recommendations)
    log(f"    Comparison table: {paths['csv']}")
    log(f"    Comparison plot: {paths['plot']}")
    log(f"    Confusion matrices: {config.CONFUSION_MATRICES_PATH}")
    log(f"    Tuned models: {paths['models_dir']}")
    log(f"    Report: {report_path}")

    total_time = time.perf_counter() - pipeline_start
    log("\n" + "=" * 60)
    log("CLASSICAL MODELS VERIFICATION COMPLETE!")
    log("=" * 60)
    log(f"Total execution time: {total_time:.2f} seconds")
    log("All 3 models trained, tuned, and compared!")

    return {
        "models": trained_models,
        "training_info": training_info,
        "test_results": results,
        "test_comparison": test_df,
        "tuned_models": tuned_models,
        "tuned_results": tuned_results,
        "best_params": best_params,
        "tuned_comparison": tuned_df,
        "recommendations": recommendations,
        "report_path": report_path,
        "total_time": total_time,
    }


if __name__ == "__main__":
    run_complete_pipeline()
