import pandas as pd
import numpy as np

from src.agents.feature_pruning import prune_features_binary_classification

def main():
    np.random.seed(42)

    # Number of samples
    n = 200

    # A strong predictive feature
    feature_good = np.random.binomial(1, 0.9, size=n)  # Mostly 1s

    # Label is exactly the same as the good feature (perfect predictor)
    label = feature_good.copy()
    feature_stupid = 1 - label
    noise = np.random.binomial(1, 0.1, size=n)  # Add noise
    feature_good = (feature_good + noise) % 2  # Add noise to the good feature
    feature_stupid = np.random.binomial(1, 0.5, size=n)

    # Combine into DataFrame
    df = pd.DataFrame({
        "feature_good": feature_good,
        "feature_stupid": feature_stupid,
        "label": label
    })

    pruned_features = prune_features_binary_classification(
        X=df.drop(columns=["label"]),
        y=df["label"],
        eval_metric="roc_auc",
        time_limit_per_split=60  # 60 seconds per fold
    )
    df = df[pruned_features]


if __name__ == "__main__":
    main()
