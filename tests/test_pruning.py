import pandas as pd
import numpy as np
from collections import Counter
from src.agents.feature_pruning import prune_features_binary_classification
from collections import Counter
from itertools import chain


def generate_data(seed, n=200):
    np.random.seed(seed)

    # A strong predictive feature
    feature_good = np.random.binomial(1, 0.9, size=n)

    # Label = perfect copy of good feature
    label = feature_good.copy()
    noise = np.random.binomial(1, 0.1, size=n)
    feature_good = (feature_good + noise) % 2

    # Useless/noisy feature
    feature_stupid = np.random.binomial(1, 0.5, size=n)

    return pd.DataFrame(
        {"feature_good": feature_good, "feature_stupid": feature_stupid, "label": label}
    )


def main(verbosity=True):
    seeds = range(10)
    results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        df = generate_data(seed)
        selected_features_per_split = prune_features_binary_classification(
            X=df.drop(columns=["label"]),
            y=df["label"],
            eval_metric="roc_auc",
            time_limit_per_split=30,
        )
        feature_counts = Counter(chain.from_iterable(selected_features_per_split))
        num_splits = len(selected_features_per_split)
        selected_features = [
            f for f, count in feature_counts.items() if count >= (num_splits // 2 + 1)
        ]
        print(f"Selected features: {selected_features}")
        results.append(selected_features)

    all_selected = [f for run in results for f in run]
    counts = Counter(all_selected)
    print("\n=== Feature Selection Frequency Across Seeds ===")
    for feature, count in counts.items():
        print(f"{feature}: {count}/10")


if __name__ == "__main__":
    main()
