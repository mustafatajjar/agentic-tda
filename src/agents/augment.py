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

    def add_column(
        self,
        df: pd.DataFrame,
        domain_context: dict,
        augmentation_goal: str = None,
        aprompt: str = None,
        num_columns_to_add: int = 10,  # <-- Added argument with default
    ) -> Tuple[pd.DataFrame, str, list]:
        """
        Adds multiple meaningful new columns to the DataFrame based on domain context.
        """
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

        sparql_result, purpose, expected_columns = (
            "",
            "",
            "",
        )  # self.sparql_prompting(df, domain_context)

        # Format the prompt with actual values, including num_columns_to_add
        prompt = prompt_template.format(
            primary_domain=domain_context.get("primary_domain", "Unknown"),
            column_descriptions=json.dumps(
                domain_context.get("column_descriptions", {}), indent=2
            ),
            table_summary=formatted_summary,
            sample_row=json.dumps(sample_row, indent=2),
            augmentation_section=augmentation_section,
            sparql_result=sparql_result,
            purpose=purpose,
            expected_columns=expected_columns,
            num_columns_to_add=num_columns_to_add,  # <-- Added to prompt
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
            return augmented_df, prompt, suggestions
        except Exception as e:
            print(f"Failed to add columns: {str(e)}")
            return df, prompt, suggestions

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
        cmd = [
            "grasp",
            "--config", "configs/single_kg.yaml",
            "--question", query
        ]

        env = {
            **os.environ,
            "MODEL": "openai/gpt-4.1",
            "FN_SET": "search_extended",
            "KG": "wikidata",
        }

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        os.chdir(prev_cwd)
        result = json.loads(result.stdout)
        return result["result"]

    def sparql_prompting(self, df: pd.DataFrame, domain_context: dict):
        sample_row = df.sample(1).to_dict(orient="records")[0]

        # Prepare data
        summary_dict = summarize_dataframe(df).reset_index()
        summary_dict = summary_dict.astype(str).fillna("null").to_dict(orient="records")

        # Pre-format JSON
        formatted_summary = json.dumps(summary_dict, indent=2)

        prompt = f"""
        You are given a table with multiple columns. Your task is:
        Generate a prompt for an LLM that will make SPARQL queries with GRASP.
        Query precise information about one column in the table and ask about the corresponding information.
        That is, use a relation that you assume to be relevant to the column you are querying, and query this information only.
        Plase find the unique values in the corresponding column of the dataset summary.
        Note that the sparql query is not able to access the dataset summary, to you need to provide the unique values in the prompt.

        ─────────────────────── DOMAIN CONTEXT ─────────────────────────
        {domain_context}

        ─────────────────────── DATASET SUMMARY ────────────────────────
        {formatted_summary}

        ──────────────────────── FIRST ROW SAMPLE ──────────────────────
        {sample_row}

        ________________________ EXAMPLES _______________________________

        "Return the nationalities of Barack Obama, Angela Merkel, Emmanuel Macron and Viktor Orban."
        "Query the language spoken in the countries Burkina Faso, Uzbekistan, Argentina and Laos."
        "Give the population of the following cities: New York, Los Angeles, Chicago, Houston, Phoenix."

        Return only this JSON object (no markdown fences):

        {{
            "prompt": "<your prompt here>",
            "column": "Column you have queried information about" (str)
        }}
        """
        print(prompt)

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        response = json.loads(response.choices[0].message.content)
        prompt = response["prompt"]
        column = response["column"]
        response = self.get_grasp_response(prompt)

        return response, column
