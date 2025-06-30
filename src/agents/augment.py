from dotenv import load_dotenv
import googlesearch
import json
from openai import OpenAI
import os
from typing import List, Dict, Union

import numpy as np
import pandas as pd

load_dotenv()


class AugmentAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def add_column(
        self, df: pd.DataFrame, domain_context: dict, augmentation_goal: str = None
    ) -> pd.DataFrame:
        """
        Adds one meaningful new column to the DataFrame based on domain context.

        Args:
            df: Original DataFrame
            domain_context: Output from DomainAgent.analyze()
            augmentation_goal: Optional specific goal for the augmentation

        Returns:
            DataFrame with one new column added
        """
        sample_row = df.sample(1).to_dict(orient="records")[0]

        augmentation_section = (
            f"=== AUGMENTATION GOAL ===\n{augmentation_goal}"
            if augmentation_goal
            else ""
        )

        prompt = f"""
        You are a data augmentation assistant. Based on the following domain context and sample data,
        suggest ONE meaningful new column that could be derived or calculated from the existing data.

        === DOMAIN CONTEXT ===
        Primary Domain: {domain_context.get('primary_domain', 'Unknown')}
        Column Descriptions: {json.dumps(domain_context.get('column_descriptions', {}), indent=2)}

        === SAMPLE ROW ===
        {json.dumps(sample_row, indent=2)}

        {augmentation_section}

        Provide:
        1. "name": The column name (make it clear and descriptive)
        2. "description": What the column represents
        3. "generation_method": A SINGLE pandas operation to create it
                            (must work when applied to the entire DataFrame)
        4. "value_example": Example value based on the sample row

        Return a JSON object with these four elements.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        suggestion = json.loads(response.choices[0].message.content)

        # Apply the augmentation
        try:
            # Create a copy to avoid modifying the original
            augmented_df = df.copy()
            # Execute the generation method
            exec_globals = {
                "df": augmented_df,
                "np": np,
                "pd": pd,
                "__builtins__": {  # Only allow safe built-ins
                    "int": int,
                    "float": float,
                    "range": range,
                    "list": list,
                    "dict": dict,
                },
            }
            exec(suggestion["generation_method"], exec_globals)
            return augmented_df
        except Exception as e:
            print(f"Failed to add column '{suggestion.get('name', '')}': {str(e)}")
            return df

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
        You are a data augmentation assistant. Based on the following domain context and sample data,
        suggest ONE meaningful augmentation that can be derived from the existing data.
        You are free to add external knowledge that you are sure about.
        Use one of the following two augmentations:

        Binning:
        replace numerical values by the mean value the interval, the size of which is bin_size
        input_columns should be a list of one column, the column to bin, the output_column is the column to include

        Mapping:
        return a dictionary, that maps from the values of the column values to a corresponding value
        input_columns should be a list of the input columns, unique values will be sent back for another query


        === DOMAIN CONTEXT ===
        Primary Domain: {domain_context.get('primary_domain', 'Unknown')}
        Column Descriptions: {json.dumps(domain_context.get('column_descriptions', {}), indent=2)}

        === SAMPLE ROW ===
        {json.dumps(sample_row, indent=2)}

        {augmentation_section}

        Provide:
        1. "method": Choose one out of "Binning" and "Mapping" (str)
        2. "input_columns": List of input columns (list[str])
        3. "output_column": Name of the new column (str)
        4. "bin_size": Size of a bin in case of binning (float)

        Return a JSON object with these five elements.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        suggestion = json.loads(response.choices[0].message.content)
        if suggestion["method"] == "Binning":
            return suggestion

    def make_augmentation(self, df: pd.DataFrame, augmentation: dict) -> pd.DataFrame:
        method = augmentation["method"]
        bin_size = augmentation["bin_size"]
        input_columns = augmentation["input_columns"]
        output_column = augmentation["output_column"]
        if method == "Binning":
            if len(input_columns) != 1:
                print("Invalid amount of columns specified for binning. No changes made.")
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
            pass
        else:
            print("Invalid augmentation specified. No changes made.")
        return df
