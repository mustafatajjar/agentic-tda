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


def test_mapping():
    aa = AugmentAgent()
    df = pd.DataFrame({
        "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Chicago", "Houston"],
        "population": [8_336_817, 3_979_576, 2_693_976, 2_320_268, 1_680_992, 2_700_000, 2_300_000]
    })
    mapping = {
        ("New York",): "NY",
        ("Los Angeles",): "CA",
        ("Chicago",): "IL",
        ("Houston",): "TX",
        ("Phoenix",): "AZ",
    }
    augmentation = {
        "method": "Mapping",
        "input_columns": ["city"],
        "output_column": "state",
        "bin_size": 0,
        "mapping": mapping
    }
    print("Original DataFrame:", df)
    df = aa.make_augmentation(df, augmentation)
    print("Augmented DataFrame:", df)


def test_sparql_prompting():
    aa = AugmentAgent()

    df = pd.DataFrame([
        ['Toyota', 'Corolla', 2020, 20000],
        ['Ford', 'Mustang', 2019, 26000],
        ['Honda', 'Civic', 2021, 22000],
        ['Tesla', 'Model 3', 2022, 35000],
        ['BMW', 'X5', 2018, 45000]
    ], columns=['Make', 'Model', 'Year', 'Price_USD'])


    domain_context = {
        "primary_domain": "Cars",
        "column_descriptions": {
            "Make": "Name of the car brand",
            "Model": "Car Model",
            "Year": "Year the car was built",
            "Price_USD": "Price in US Dollars"
        }
    }

    print(aa.sparql_prompting(df, domain_context))


if __name__ == "__main__":
    test_sparql_prompting()
    # test_binning()
    # test_mapping()
