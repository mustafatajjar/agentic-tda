import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_arff_to_dataframe(file_path):
    """
    Load ARFF file into pandas DataFrame
    
    Parameters:
    - file_path: path to the ARFF file
    
    Returns:
    - pandas DataFrame
    """
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Decode byte strings to regular strings if needed
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    
    for col in str_df:
        df[col] = str_df[col]
        
    return df

def evaluate(table):
    # preprocess data
    df = load_arff_to_dataframe("data/")
    X = df.drop(columns=["binaryClass"])
    y = df["binaryClass"]
    

    # fit model
    model = RandomForestClassifier()
    
    # predict

    # calculate score
    pass
