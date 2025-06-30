import pandas as pd

from src.agents.augment import AugmentAgent


def test_binning():
    aa = AugmentAgent()
    df = pd.DataFrame({
        "age": [15, 22, 37, 45, 53, 67, 80],
        "income": [5000, 15000, 32000, 45000, 60000, 72000, 85000],
        "score": [3.2, 4.5, 6.7, 5.5, 8.8, 7.1, 9.0]
    })
    augmentation = {
        "method": "Binning",
        "input_columns": ["score"],
        "output_column": "binned_score",
        "bin_size": 3
    }
    print("Original DataFrame:", df)
    df = aa.make_augmentation(df, augmentation)
    print("Augmented DataFrame:", df)

if __name__ == "__main__":
    test_binning()