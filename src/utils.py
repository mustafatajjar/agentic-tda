import pandas as pd


def summarize_dataframe(df):
    """Generate a summary of the DataFrame: dtype, uniques, missing, stats."""
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'unique_values': df.nunique(),
        'missing_values': df.isnull().sum(),
    })

    numeric_cols = df.select_dtypes(include='number').columns
    summary.loc[numeric_cols, 'mean'] = df[numeric_cols].mean()
    summary.loc[numeric_cols, 'min'] = df[numeric_cols].min()
    summary.loc[numeric_cols, 'max'] = df[numeric_cols].max()
    summary.loc[numeric_cols, 'std'] = df[numeric_cols].std()

    return summary