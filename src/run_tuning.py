from dotenv import load_dotenv
from src.Tuning.tuning import run_experiment
from promptolution.utils import ExperimentConfig
from src.utils import arff_to_dataframe, extract_arff_metadata
from src.agents.feature_pruning import prune_features_binary_classification

load_dotenv()  # Load API keys

import os
import random
import pandas as pd

def load_init_prompts(folder_path):
    init_prompt = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                init_prompt.append(f.read())
    return init_prompt

def create_xy_dataframe(df, n_rows=100):
    columns = list(df.columns)
    data = []
    for _ in range(n_rows):
        y_col = random.choice(columns)
        x_cols = [col for col in columns if col != y_col]
        data.append({'x': x_cols, 'y': y_col})
    return pd.DataFrame(data)

# Usage:
# folder = os.path.join(os.path.dirname(__file__), "prompts", "augment")
# init_prompt = load_init_prompts(folder)

task_description = (
    "The task is to add new columns to a DataFrame based on the provided context with the goal of imroving the evaluation score. "
)

def main():
    # Load initial prompts
    folder = os.path.join(os.path.dirname(__file__), "agents", "prompts", "augment")
    init_prompts = load_init_prompts(folder)

    task_description = (
    "The task is to add new columns to a DataFrame based on the provided context with the goal of improving the evaluation score. "
    )
    # Run the tuning process with custom agents
    api_key = os.getenv("OPENAI_API_KEY")

    config = ExperimentConfig(
        optimizer="capo",
        task_description=task_description,
        prompts=init_prompts,
        n_steps=10,
        api_url="https://api.openai.com/v1",
        model_id="gpt-4.1-mini",
        api_key=api_key,
        n_subsamples=30,
    )
    print('hi')

    arff_file_path = "./data/dataset_31_credit-g.arff"

    metadata = extract_arff_metadata(arff_file_path)
    df = arff_to_dataframe(arff_file_path)
    xy_df = create_xy_dataframe(df)
    print(xy_df.head())
    prompts = run_experiment(xy_df, config)

    print("Prompts generated:", prompts)

    

if __name__ == "__main__":
    main()

