import pandas as pd
import numpy as np

from src.agents.feature_pruning import prune_features_binary_classification

def main():
    np.random.seed(42)

    # Number of samples
    n = 200

    # A strong predictive feature
    feature_good = np.random.binomial(1, 0.7, size=n)  # Mostly 1s

    # Label is exactly the same as the good feature (perfect predictor)
    label = feature_good.copy()

    # Add a misleading column (stupid feature):
    # It's the inverse of the label with some noise — a trap for the model
    feature_stupid = 1 - label
    print("features_stupid")
    print(feature_stupid)
    noise = np.random.binomial(1, 0.1, size=n)  # Add noise
    print("noise")
    print(noise)
    feature_stupid = (feature_stupid + noise) % 2  # Flip a few bits
    print("feature_stupid after noise")
    print(feature_stupid)

    # Combine into DataFrame
    df = pd.DataFrame({
        "feature_good": feature_good,
        "feature_stupid": feature_stupid,
        "label": label
    })
    
    print("Initial DataFrame:")
    print(df.head())

    print()
    pruned_features = prune_features_binary_classification(
        X=df.drop(columns=["label"]),
        y=df["label"],
        eval_metric="roc_auc",
        time_limit_per_split=60  # 60 seconds per fold
    )
    df = df[pruned_features]

    print("Pruned DataFrame:")
    print(df.head())


if __name__ == "__main__":
    main()
