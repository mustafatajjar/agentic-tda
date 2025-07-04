import json
from openai import OpenAI
import os
import pandas as pd
from typing import Tuple

from src.utils import summarize_dataframe


# --------- Agent ---------
class DomainAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def analyze(self, df: pd.DataFrame, arff_metadata: str = "") -> Tuple[dict, str]:
        summary_df = summarize_dataframe(df)
        summary_dict = summary_df.reset_index()
        summary_dict = summary_dict.astype(str).fillna("null").to_dict(orient="records")
        sample_row = df.sample(1).to_dict(orient="records")[0]

        prompt = f"""
        You are a domain analysis assistant. Use the following ARFF file metadata and table summary
        to infer the dataset's domain and structure.

        === ARFF METADATA ===
        {arff_metadata}

        === COLUMN SUMMARY ===
        (dtype, unique values, missing, mean/min/max/std if numeric):
        {json.dumps(summary_dict, indent=2)}

        === SAMPLE ROW ===
        {json.dumps(sample_row, indent=2)}

        Return a JSON object with:
        1. "primary_domain": Broad category (e.g., "healthcare", "ecommerce").
        2. "column_descriptions": Brief explanations for each column's purpose.
        3. "important_metadata": A filtered summary of only the key information from the ARFF metadata.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content), prompt
