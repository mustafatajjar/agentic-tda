from dotenv import load_dotenv
import googlesearch
import json
from openai import OpenAI
import os
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from typing import Tuple

load_dotenv()


class AugmentAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def add_column(
        self, df: pd.DataFrame, domain_context: dict, augmentation_goal: str = None
    ) -> Tuple[pd.DataFrame, str, dict]:
        """
        Adds one meaningful new column to the DataFrame based on domain context.
        """
        # Load prompt from file
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', 'reasoning_type.txt')
        with open(prompt_path, 'r') as file:
            prompt_template = file.read()
        
        sample_row = df.sample(1).to_dict(orient="records")[0]

        augmentation_section = (
            f"=== AUGMENTATION GOAL ===\n{augmentation_goal}"
            if augmentation_goal
            else ""
        )

        # Format the prompt with actual values
        prompt = prompt_template.format(
            primary_domain=domain_context.get('primary_domain', 'Unknown'),
            column_descriptions=json.dumps(domain_context.get('column_descriptions', {}), indent=2),
            sample_row=json.dumps(sample_row, indent=2),
            augmentation_section=augmentation_section
        )

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        suggestion = json.loads(response.choices[0].message.content)
        
        print("\nGPT Suggestion:")
        print(json.dumps(suggestion, indent=2))
        print(f"\nGeneration method to execute: {suggestion['generation_method']}")
        
        try:
            augmented_df = df.copy()
            if suggestion["name"] in augmented_df.columns:
                print(f"Column '{suggestion['name']}' already exists - skipping")
                return augmented_df
            exec_globals = {
                "df": augmented_df,
                "np": np,
                "pd": pd,
                "__builtins__": {
                    "int": int,
                    "float": float,
                    "range": range,
                    "list": list,
                    "dict": dict,
                },
            }
            exec(suggestion["generation_method"], exec_globals)
            return augmented_df, prompt, suggestion
        except Exception as e:
            print(f"Failed to add column '{suggestion.get('name', '')}': {str(e)}")
            return df, prompt, suggestion

    def mapping_binning_augment(
        self, df: pd.DataFrame, domain_context: dict, augmentation_goal: str = None
    ) -> pd.DataFrame:
        augmentation = self.get_augmentation(
            df=df, domain_context=domain_context, augmentation_goal=augmentation_goal
        )
        augmented_df = self.make_augmentation(df=df, augmentation=augmentation)
        return augmented_df

    def get_augmentation(
        self, df: pd.DataFrame, domain_context: dict, augmentation_goal: str = None
    ) -> dict:
        """
        Computes an augmentation, chosen from Mapping and Binning

        Args:
            df: Original DataFrame
            domain_context: Output from DomainAgent.analyze()
            augmentation_goal: Optional specific goal for the augmentation

        Returns:
            Augmentation as JSON object from LLM
        """
        sample_row = df.sample(1).to_dict(orient="records")[0]

        augmentation_section = (
            f"=== AUGMENTATION GOAL ===\n{augmentation_goal}"
            if augmentation_goal
            else ""
        )

        prompt = f"""
        You are a data augmentation assistant. Given a dataset and domain context, your task is to suggest one meaningful new column
        that can be derived by applying a transformation to one or more existing columns.

        There are two transformation types:
        - "Mapping": Combine categorical input columns into a new categorical value.
        - "Binning": Group numerical values into bins and replace with the mean of the bin.

        Rules:
        - For Mapping: Choose 1 or 2 categorical input columns with <= 10000 value combinations.
        - The output_column must be a new, descriptive column name.

        === DOMAIN CONTEXT ===
        Primary Domain: {domain_context.get("primary_domain", "Unknown")}
        Column Descriptions:
        {json.dumps(domain_context.get("column_descriptions", {}), indent=2)}

        === SAMPLE ROW ===
        {json.dumps(sample_row, indent=2)}

        Respond with a JSON object:
        {{
        "method": "Mapping",
        "input_columns": ["col1", "col2", ...],
        "output_column": "name_of_new_column",
        "bin_size": null  // leave null for Mapping
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        suggestion = json.loads(response.choices[0].message.content)
        method = suggestion["method"]
        if method == "Binning":
            return suggestion
        elif method == "Mapping":
            # Compute a dictionary with unique values for the chosen columns
            columns = suggestion["input_columns"]
            unique_values = dict()
            for column in columns:
                unique_values[column] = df[column].unique().tolist()

            prompt = f"""
            You are a data‑augmentation assistant.
            Given a set of input columns and *every* combination of their unique values,
            create a mapping that assigns a descriptive category to **each** combination.

            ### Requirements
            1. Cover **every combination** shown in “UNIQUE VALUE COMBINATIONS”.
            2. The response **must be valid JSON** (no Markdown fences, no comments).
            3. Use this exact schema:

            {{
            "output_column": "name_of_new_column",
            "mapping": [
                {{
                "inputs": ["val_for_col1", "val_for_col2", ...],  // same order as input_columns
                "value":   "derived_category"                     // a string label
                }},
                ...
            ]
            }}

            ### Example
            If `input_columns = ["color", "size"]` and the combinations are  
            `[["red", "small"], ["blue", "large"]]`, a valid response is:

            {{
            "output_column": "color_size_category",
            "mapping": [
                {{ "inputs": ["red",  "small"], "value": "category1" }},
                {{ "inputs": ["blue", "large"], "value": "category2" }}
            ]
            }}

            ### INPUT COLUMNS
            {json.dumps(columns)}

            ### UNIQUE VALUE COMBINATIONS
            {json.dumps(unique_values, indent=2)}

            Return **only** the JSON object that follows the schema above.
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            suggestion = json.loads(response.choices[0].message.content)
            suggestion["input_columns"] = columns
            suggestion["method"] = method
            mapping = suggestion["mapping"]
            mapping = {tuple(item["inputs"]): item["value"] for item in mapping}
            suggestion["mapping"] = mapping
            return suggestion

    def make_augmentation(self, df: pd.DataFrame, augmentation: dict) -> pd.DataFrame:
        print("AUGMENTATION")
        print(augmentation)
        method = augmentation["method"]
        input_columns = augmentation["input_columns"]
        output_column = augmentation["output_column"]
        if method == "Binning":
            bin_size = augmentation["bin_size"]
            if len(input_columns) != 1:
                print(
                    "Invalid amount of columns specified for binning. No changes made."
                )
                return df
            column = input_columns[0]
            values = df[column]

            num_bins = int(np.ceil((np.max(values) - np.min(values)) / bin_size)) + 1
            bins = np.linspace(np.min(values), np.max(values), num_bins)
            df[output_column] = pd.cut(df[column], bins=bins, include_lowest=True)
            intervals = df[output_column].cat.categories
            midpoints = intervals.left + (intervals.right - intervals.left) / 2
            interval_to_mid = dict(zip(intervals, midpoints))
            df[output_column] = df[output_column].map(interval_to_mid)
        elif method == "Mapping":
            print("INPUT COLUMNS")
            print(input_columns)
            mapping = augmentation["mapping"]
            print("MAPPING")
            print(mapping)
            map_ser = pd.Series(mapping)
            map_ser.index = pd.MultiIndex.from_tuples(
                map_ser.index, names=input_columns
            )

            tuples = pd.MultiIndex.from_frame(df[input_columns])

            df[output_column] = map_ser.reindex(tuples).values
        else:
            print("Invalid augmentation specified. No changes made.")
        return df
