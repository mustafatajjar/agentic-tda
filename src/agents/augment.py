import googlesearch
from openai import OpenAI
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import json


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

    def add_column_modes(
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
