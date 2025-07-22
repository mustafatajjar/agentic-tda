from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import requests
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
        self, df: pd.DataFrame, domain_context: dict, augmentation_goal: str = None, aprompt: str = None
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

        sparql_result, purpose, expected_columns = "", "", "" # self.sparql_prompting(df, domain_context)

        # Format the prompt with actual values
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
            expected_columns=expected_columns
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
            mapping = augmentation["mapping"]
            map_ser = pd.Series(mapping)
            map_ser.index = pd.MultiIndex.from_tuples(
                map_ser.index, names=input_columns
            )

            tuples = pd.MultiIndex.from_frame(df[input_columns])

            df[output_column] = map_ser.reindex(tuples).values
        else:
            print("Invalid augmentation specified. No changes made.")
        return df

    def get_sparql_response(self, query):
        endpoint = "https://qlever.cs.uni-freiburg.de/api/wikidata"
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/sparql-query"},
            data=query
        )

        # Check result
        if response.ok:
            result = response.json()
            return result["results"]["bindings"]
        else:
            print("Query failed:", response.status_code, response.text)
            return None

    def sparql_prompting(self, df: pd.DataFrame, domain_context: dict):
        sample_row = df.sample(1).to_dict(orient="records")[0]
        
        # Prepare data
        summary_dict = summarize_dataframe(df).reset_index()
        summary_dict = summary_dict.astype(str).fillna("null").to_dict(orient="records")
        
        # Pre-format JSON
        formatted_summary = json.dumps(summary_dict, indent=2)

        # prompt = f"""
        # You are a knowledge graph expert writing SPARQL 1.1 queries for the QLever Wikidata endpoint (https://qlever.cs.uni-freiburg.de).

        # Your task is to gather knowledge about entities in the given table by querying Wikidata. You should generate a SPARQL query that explores **one entity** from the table and retrieves its direct relationships via predicates.

        # ─────────────────────── DOMAIN CONTEXT ─────────────────────────
        # {domain_context}

        # ─────────────────────── DATASET SUMMARY ────────────────────────
        # {formatted_summary}

        # ──────────────────────── FIRST ROW SAMPLE ──────────────────────
        # {sample_row}

        # Instructions:
        # - You MUST refer to only **one label** from the table (typically the subject).
        # - Use `rdfs:label` with language filtering for English (e.g., `"Madrid"@en`).
        # - Do NOT use both a subject and object label in the query.
        # - Focus on finding predicates and connected entity labels related to the selected subject label.
        # - Use only the `rdfs:label` mechanism to refer to entities (do not use Q-IDs).
        # - Use `OPTIONAL` blocks to retrieve human-readable labels (`?predicateLabel`, `?objectLabel`).
        # - Do NOT include explanations or comments outside the query or JSON.

        # Use these prefixes exactly:

        # PREFIX wd:   <http://www.wikidata.org/entity/>
        # PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
        # PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        # ─────────────── EXAMPLE ───────────────
        # To retrieve the relationships for the entity labeled "Madrid":

        # ```sparql
        # PREFIX wd:   <http://www.wikidata.org/entity/>
        # PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
        # PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        # SELECT DISTINCT ?predicate (COUNT(?object) AS ?count) ?predicateLabel ?objectLabel WHERE {{
        #     ?subject rdfs:label "Madrid"@en .
        #     ?subject ?predicate ?object .

        #     # Convert direct predicate (wdt:) to entity (wd:)
        #     BIND(IRI(REPLACE(STR(?predicate), "prop/direct/", "entity/")) AS ?predicateEntity)

        #     OPTIONAL {{
        #         ?predicate rdfs:label ?predicateLabel .
        #         FILTER(LANG(?predicateLabel) = "en")
        #     }}
        #     OPTIONAL {{
        #         ?object rdfs:label ?objectLabel .
        #         FILTER(LANG(?objectLabel) = "en")
        #     }}
        # }}
        # GROUP BY ?count DESC
        # LIMIT 100

        # ───────────────────── TASK ─────────────────────

        # Return only this JSON object (no markdown fences):

        # {{
        #     "sparql_query": "<your generated SPARQL query here>"
        #     "input_label": "the table content you were looking to query for, e.g., 'Madrid'"
        # }}
        # """


        # response = self.client.chat.completions.create(
        #     model="gpt-4.1-mini",
        #     messages=[{"role": "user", "content": prompt}],
        #     response_format={"type": "json_object"},
        # )
        # response = json.loads(response.choices[0].message.content)
        # query = response["sparql_query"]
        # sparql_response = self.get_sparql_response(query)



        # prompt = f"""
        # You are an expert SPARQL engineer working with the QLever Wikidata endpoint (https://qlever.cs.uni-freiburg.de).

        # You have just run a discovery query to find all predicates that connect two entities identified only by their `rdfs:label`. Now you will use the results of that query to construct a **new query** that retrieves meaningful information involving those predicates.
        # Please make a query with all entities from the column of the previous query input and relate them to a sensible predicate.

        # ────────────────────── PREVIOUS DISCOVERY ──────────────────────
        # The previous query returned predicates and objects that are connected to the entities in the table.

        # Here is the JSON response from the previous query:
        # {json.dumps(sparql_response, indent=2)}

        # What was looked for in the previous query:
        # {response["input_label"]}

        # ─────────────────────── DOMAIN CONTEXT ─────────────────────────
        # {domain_context}

        # ─────────────────────── DATASET SUMMARY ────────────────────────
        # {formatted_summary}

        # ──────────────────────── FIRST ROW SAMPLE ──────────────────────
        # {sample_row}

        # ─────────────────── CONSTRAINTS FOR QUERY ──────────────────────
        # - Use only the predicate URIs from the previous query result.
        # - Use `rdfs:label` to access readable names, with:
        #     FILTER(LANG(?label) = "en")
        # - Do not use SERVICE wikibase:label.
        # - Avoid guessing P-IDs — only reuse the predicates shown above.
        # - Do not include unrelated PREFIXes or unused namespaces.

        # ────────────────────────── EXAMPLE ────────────────────────────

        # PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        # SELECT ?entity ?entityLabel ?predicate ?predicateLabel ?connectedEntity ?connectedEntityLabel WHERE {{
        # VALUES ?entityLabel {{ "Entity1"@en "Entity2"@en "Entity3"@en }}  # Replace with your list of entity labels

        # ?entity rdfs:label ?entityLabel .
        # FILTER(LANG(?entityLabel) = "en")

        # ?entity ?predicate ?connectedEntity .

        # OPTIONAL {{
        #     ?predicate rdfs:label ?predicateLabel .
        #     FILTER(LANG(?predicateLabel) = "en")
        # }}

        # OPTIONAL {{
        #     ?connectedEntity rdfs:label ?connectedEntityLabel .
        #     FILTER(LANG(?connectedEntityLabel) = "en")
        # }}
        # }}
        # LIMIT 1


        # ────────────────────────── TASK ───────────────────────────────

        # Now, write a SPARQL query that explores one or more of the discovered predicates to enrich the table. It should use only predicates from the discovery result above and return readable English labels.

        # Return only this JSON object (no markdown fences):

        # {{
        #     "sparql_query": "<your full query here>",
        #     "purpose": "Describe the predicate relationships between entities based on previous discovery",
        #     "expected_columns": ["entity", "predicateLabel", "connectedEntityLabel"]
        # }}
        # """

        prompt = f"""
        You are given a table with multiple columns. Your task is:

        - Pick one column from the table whose values you will use.
        - Take all strings in that column.
        - Construct a SPARQL SELECT query that looks for triples where:
        - The subject's label contains one of the strings from the chosen column (using CONTAINS with LCASE for case-insensitive matching).
        - The predicate's label contains a meaningful keyword indicating the relation you want to find (also using CONTAINS with LCASE).
        - Return a single SPARQL query that covers all values from the column, e.g., by using VALUES or multiple FILTER conditions.
        - Return also the name of the column you used.

        Example output format:

        Column used: <column_name>

        SPARQL Query:
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?subject ?property ?object WHERE {{
        VALUES ?searchString {
            "string1"
            "string2"
            "string3"
        }

        ?subject rdfs:label ?subjectLabel .
        FILTER(LANG(?subjectLabel) = "en")
        FILTER(CONTAINS(LCASE(?subjectLabel), LCASE(?searchString)))

        ?subject ?property ?object .

        ?property rdfs:label ?propertyLabel .
        FILTER(CONTAINS(LCASE(?propertyLabel), "<relation keyword>"))

        ?object rdfs:label ?objectLabel .
        FILTER(LANG(?objectLabel) = "en")
        }}
        LIMIT 50


        Notes:
        - Use lowercase matching with CONTAINS for flexibility.
        - Make sure the query is syntactically valid.

        Return only this JSON object (no markdown fences):

        {{
            "sparql_query": "<your full query here>",
            "column": "Column you have used to construct the query"
        }}
        """


        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        response = json.loads(response.choices[0].message.content)
        query = response["sparql_query"]
        print(query)
        column = response["column"]
        # purpose = response["purpose"]
        # expected_columns = response["expected_columns"]
        response = self.get_sparql_response(query)
        print(response)
        return response, column
        # return response, purpose, expected_columns
