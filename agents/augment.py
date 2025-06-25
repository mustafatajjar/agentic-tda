import googlesearch  # pip install google
from openai import OpenAI
import os
from typing import List, Dict
import pandas as pd

class AugmentAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def suggest_columns(self, domain_context: Dict, df: pd.DataFrame) -> List[str]:
        """Use Google + LLM to propose new columns based on domain."""
        # Step 1: Get domain-specific augmentation strategies via Google
        search_query = f"Common columns to augment {domain_context['primary_domain']} dataset"
        search_results = self._google_search(search_query)
        
        # Step 2: Ask LLM to synthesize column ideas
        prompt = self._build_augment_prompt(domain_context, df, search_results)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return self._parse_llm_response(response.choices[0].message.content)

    def _google_search(self, query: str, num_results: int = 3) -> List[str]:
        """Fetch top Google search snippets (requires internet)."""
        try:
            return [result for result in googlesearch.search(query, num_results=num_results)]
        except:
            return []  # Fallback if search fails

    def _build_augment_prompt(self, context: Dict, df: pd.DataFrame, search_results: List[str]) -> str:
        """Construct a prompt mimicking data scientist thinking."""
        return f"""
        You are a data scientist analyzing a {context['primary_domain']} dataset. Here's context:
        - Current columns: {list(df.columns)}
        - Column descriptions: {context.get('column_descriptions', {})}
        - Domain research: {search_results[:2]}  # Top 2 Google results

        Suggest 5 new columns that would add analytical value. For each:
        1. Explain why it's useful.
        2. Specify the data type (e.g., float, categorical).
        3. Provide a sample calculation if derived from existing columns.

        Format as a numbered list:
        1. [Column Name]: [Description]. Type: [dtype]. Example: [formula].
        """

    def _parse_llm_response(self, response: str) -> List[str]:
        """Extract column names from LLM response."""
        return [line.split(":")[0].strip() for line in response.split("\n") if line.strip()]