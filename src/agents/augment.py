from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import requests
import subprocess
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from typing import Tuple

from src.utils import summarize_dataframe

load_dotenv()


class AugmentAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.latest_added_columns = []
        self.queried_columns = []
        self.queried_relations = []

    def add_column(
        self,
        df: pd.DataFrame,
        domain_context: dict,
        history_responses: list = None,
        selected_features_history: list = None,
        augmentation_goal: str = None,
        aprompt: str = None,
        num_columns_to_add: int = 10,
        target_column: str = None,
    ) -> Tuple[pd.DataFrame, str, list]:
        """
        Adds multiple meaningful new columns to the DataFrame based on domain context.
        """
        # Prepare augmentation history string
        history_responses = history_responses or []
        selected_features_history = selected_features_history or []
        augmentation_history = []
        for i, (resp, feats) in enumerate(
            zip(history_responses, selected_features_history)
        ):
            augmentation_history.append(
                f"Trial {i+1}:\nPrompt: {resp}\nSelected Features: {feats}\n"
            )
        augmentation_history_str = (
            "\n".join(augmentation_history) if augmentation_history else "None"
        )

        # Use provided prompt if given, else load from file
        if aprompt is not None:
            prompt_template = aprompt
        else:
            prompt_path = os.path.join(
                os.path.dirname(__file__), "prompts", "refined_reasoning_type.txt"
            )
            with open(prompt_path, "r") as file:
                prompt_template = file.read()

        sample_row = df.sample(1).to_dict(orient="records")[0]

        # Prepare data
        summary_dict = summarize_dataframe(df).reset_index()
        summary_dict = summary_dict.astype(str).fillna("null").to_dict(orient="records")

        # Pre-format JSON
        formatted_summary = json.dumps(summary_dict, indent=2)

        augmentation_section = (
            f"=== AUGMENTATION GOAL ===\n{augmentation_goal}"
            if augmentation_goal
            else ""
        )

        # sparql_result, column = self.sparql_prompting(df, domain_context)

        # Format the prompt with actual values, including augmentation_history
        prompt = prompt_template.format(
            primary_domain=domain_context.get("primary_domain", "Unknown"),
            column_descriptions=json.dumps(
                domain_context.get("column_descriptions", {}), indent=2
            ),
            table_summary=formatted_summary,
            sample_row=json.dumps(sample_row, indent=2),
            augmentation_section=augmentation_section,
            sparql_result='none',
            column='column',
            num_columns_to_add=num_columns_to_add,
            augmentation_history=augmentation_history_str,  # <-- Pass history here
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        suggestions = json.loads(response.choices[0].message.content)["columns"]

        try:
            augmented_df = df.copy()
            added_columns = []
            for suggestion in suggestions:
                col_name = suggestion["name"]
                # Exclude columns that use the target column in their generation method
                if target_column and target_column in suggestion.get("generation_method", ""):
                    print(f"Skipping column '{col_name}' because it uses the target column '{target_column}'.")
                    continue
                if col_name in augmented_df.columns:
                    print(f"Column '{col_name}' already exists - skipping")
                else:
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
                    added_columns.append(col_name)

            self.latest_added_columns = added_columns
            return augmented_df, prompt, suggestions, True
        except Exception as e:
            print(f"Failed to add columns: {str(e)}")
            return df, prompt, suggestions, False

    def get_sparql_response(self, query):
        endpoint = "https://qlever.cs.uni-freiburg.de/api/wikidata"
        response = requests.post(
            endpoint, headers={"Content-Type": "application/sparql-query"}, data=query
        )

        # Check result
        if response.ok:
            result = response.json()
            return result["results"]["bindings"]
        else:
            print("Query failed:", response.status_code, response.text)
            return None

    def get_grasp_response(self, query: str) -> str:
        prev_cwd = os.getcwd()
        os.chdir("grasp")
        cmd = ["grasp", "--config", "configs/single_kg.yaml", "--question", query]

        env = {
            **os.environ,
            "MODEL": "openai/gpt-4.1",
            "FN_SET": "search_extended",
            "KG": "wikidata",
        }

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        os.chdir(prev_cwd)
        if not result.stdout:
            # print("No output from GRASP command.\n" * 100)
            return ""
        result = json.loads(result.stdout)
        return result["result"]

    def sparql_prompting(self, df: pd.DataFrame, domain_context: dict):
        sample_row = df.sample(1).to_dict(orient="records")[0]

        # Prepare data
        summary_dict = summarize_dataframe(df).reset_index()
        summary_dict = summary_dict.astype(str).fillna("null").to_dict(orient="records")

        # Pre-format JSON
        formatted_summary = json.dumps(summary_dict, indent=2)

        prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", "sparql_prompt.txt"
        )
        with open(prompt_path, "r") as file:
            prompt_template = file.read()

        prompt = prompt_template.format(
            domain_context=domain_context,
            formatted_summary=formatted_summary,
            sample_row=sample_row,
            queried_columns=self.queried_columns,
            queried_relations=self.queried_relations,
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        response = json.loads(response.choices[0].message.content)
        prompt = response["prompt"]
        column = response["column"]
        relation = response["relation"]
        self.queried_columns.append(column)
        self.queried_relations.append(relation)
        response = self.get_grasp_response(prompt)

        return response, column
