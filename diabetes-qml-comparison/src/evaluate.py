#phase 4 : evaluate phase

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

def score_prediction(name, y_true, y_pred):
    """Calculates common classification metrics."""
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 score": f1_score(y_true, y_pred, zero_division=0)
    }

def compare_result(result_dict, plot=True):
    """Saves a comparison table and optionally plots a chart."""
    df = pd.DataFrame(result_dict).set_index("Model")

    # Save to CSV
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / 'comparison_metrics.csv')

    if plot:
        df.plot(kind='bar', figsize=(10, 6))
        plt.title("Model Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_plot.png')
        print(f"Comparison saved to {output_dir}")

    return df

if __name__ == "__main__": 
    print("Testing evaluation module in isolation...")
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    metrics = score_prediction("Test_Model", y_true, y_pred)
    
    results = compare_result({
        "Model": [metrics["Model"]], 
        "Accuracy": [metrics["Accuracy"]], 
        "Precision": [metrics["Precision"]], 
        "Recall": [metrics["Recall"]], 
        "F1 score": [metrics["F1 score"]]
    })

    print(results)