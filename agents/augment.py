import googlesearch
from openai import OpenAI
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Union

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
        return self._parse_llm_suggestions(response.choices[0].message.content)

    def generate_synthetic_column(self, df: pd.DataFrame, column_description: str, 
                                num_rows: int = None) -> pd.Series:
        """
        Generate synthetic data for a new column based on description and existing data.
        
        Args:
            df: The existing DataFrame
            column_description: Description of the column to generate (from suggest_columns)
            num_rows: Number of rows to generate (default: same as input DataFrame)
            
        Returns:
            pd.Series with synthetic data
        """
        if num_rows is None:
            num_rows = len(df)
            
        prompt = self._build_synthetic_data_prompt(df, column_description, num_rows)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7  # Higher temp for more creative generation
        )
        
        return self._parse_synthetic_data(response.choices[0].message.content, num_rows)

    def augment_dataframe(self, domain_context: Dict, df: pd.DataFrame, 
                        num_suggestions: int = 3) -> pd.DataFrame:
        """
        Full pipeline: suggest columns and add them with synthetic data.
        
        Args:
            domain_context: Dictionary with domain information
            df: Input DataFrame to augment
            num_suggestions: How many new columns to add
            
        Returns:
            Augmented DataFrame
        """
        augmented_df = df.copy()
        column_suggestions = self.suggest_columns(domain_context, df)
        
        for i, col_desc in enumerate(column_suggestions[:num_suggestions]):
            if i >= num_suggestions:
                break
            try:
                new_col = self.generate_synthetic_column(df, col_desc)
                col_name = col_desc.split(":")[0].strip()
                augmented_df[col_name] = new_col
            except Exception as e:
                print(f"Failed to generate column {col_desc}: {str(e)}")
        
        return augmented_df

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

    def _build_synthetic_data_prompt(self, df: pd.DataFrame, column_description: str, 
                                   num_rows: int) -> str:
        """Construct prompt for generating synthetic data."""
        return f"""
        You are a data generator for a {df.shape[0]} row dataset. Here's the context:
        - Existing columns: {list(df.columns)}
        - First row values: {df.iloc[0].to_dict()}
        
        Generate synthetic data for this new column:
        {column_description}
        
        The data should be realistic and consistent with existing columns where relevant.
        For numerical columns, provide a comma-separated list of values.
        For categorical columns, provide a comma-separated list of values in quotes.
        For datetime columns, use YYYY-MM-DD format.
        
        Return ONLY the raw data values, one per row, separated by commas. No explanation.
        Example outputs:
        - Numerical: 1.2, 3.4, 5.6
        - Categorical: "apple", "banana", "apple"
        """

    def _parse_llm_suggestions(self, response: str) -> List[str]:
        """Extract column names and descriptions from LLM response."""
        return [line.strip() for line in response.split("\n") if line.strip() and line[0].isdigit()]

    def _parse_synthetic_data(self, response: str, expected_rows: int) -> Union[pd.Series, None]:
        """Parse the LLM's synthetic data response into a pandas Series."""
        try:
            # Clean the response
            clean_response = response.strip().replace('"', '').replace("'", "")
            values = [x.strip() for x in clean_response.split(",")]
            
            if len(values) != expected_rows:
                print(f"Warning: Expected {expected_rows} values, got {len(values)}")
                # If we got fewer values than expected, pad with NaN
                if len(values) < expected_rows:
                    values += [np.nan] * (expected_rows - len(values))
                # If we got more, truncate
                else:
                    values = values[:expected_rows]
            
            # Try to convert to numerical first
            try:
                return pd.to_numeric(pd.Series(values))
            except ValueError:
                # If not numerical, return as string
                return pd.Series(values)
        except Exception as e:
            print(f"Error parsing synthetic data: {str(e)}")
            return None