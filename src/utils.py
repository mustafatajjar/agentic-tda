import arff
import pandas as pd
from scipy.io import arff as sparff


def summarize_dataframe(df):
    """Generate a summary of the DataFrame: dtype, uniques, missing, stats."""
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "unique_count": df.nunique(),
            "missing_values": df.isnull().sum(),
        }
    )
    
    # Add unique values only for non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude="number").columns
    summary["unique_values"] = [
        df[col].dropna().unique().tolist()
        if col in non_numeric_cols else None
        for col in df.columns
    ]

    
    # For numeric columns, set unique_values to "continuous values"
    numeric_cols = df.select_dtypes(include="number").columns
    summary.loc[numeric_cols, "unique_values"] = "continuous values"
    
    # Add stats for numeric columns
    summary.loc[numeric_cols, "mean"] = df[numeric_cols].mean()
    summary.loc[numeric_cols, "min"] = df[numeric_cols].min()
    summary.loc[numeric_cols, "max"] = df[numeric_cols].max()
    summary.loc[numeric_cols, "std"] = df[numeric_cols].std()
    
    return summary


def load_arff_to_dataframe(file_path):
    """
    Load ARFF file into pandas DataFrame

    Parameters:
    - file_path: path to the ARFF file

    Returns:
    - pandas DataFrame
    """
    data, _ = sparff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Decode byte strings to regular strings if needed
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8")

    return df


def arff_to_dataframe(file_path):
    """Convert ARFF to pandas DataFrame."""
    with open(file_path, "r") as f:
        arff_data = arff.load(f)
    data = pd.DataFrame(
        arff_data["data"], columns=[attr[0] for attr in arff_data["attributes"]]
    )
    return data


def extract_arff_metadata(file_path):
    """Extract metadata comments (lines starting with %) from an ARFF file."""
    comments = []
    with open(file_path, "r") as file:
        for line in file:
            if line.strip().startswith("%"):
                comments.append(line.strip("%").strip())
    return "\n".join(comments)
