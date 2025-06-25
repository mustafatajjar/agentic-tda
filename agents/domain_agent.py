import json
from openai import OpenAI
import os
import pandas as pd

import json
from openai import OpenAI
import os
import pandas as pd

class DomainAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def analyze(self, df: pd.DataFrame) -> dict:
        """Analyze table and return domain + column descriptions (no APIs)."""
        prompt = f"""
        Analyze this table structure:
        - Columns: {df.columns.tolist()}
        - Data types: {df.dtypes.to_dict()}
        - Sample row: {df.head(1).to_dict()}

        Return JSON with:
        1. "primary_domain": Broad category (e.g., "healthcare", "ecommerce").
        2. "column_descriptions": Brief explanations for each column's purpose.
        3. "potential_use_cases": Example analyses this data could support.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)  