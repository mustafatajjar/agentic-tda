import json
from openai import OpenAI
import os
import pandas as pd
from typing import Tuple

from src.utils.funcs import summarize_dataframe
from src.agents.core.agent import Agent, AgentInput, AgentOutput
from dataclasses import dataclass

@dataclass
class DomainAgentInput:
    df: pd.DataFrame
    arff_metadata: str = ""

class DomainAgent(Agent):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def run(self, input: AgentInput):
        agent_input = DomainAgentInput(**input.data)
        df = agent_input.df
        arff_metadata = agent_input.arff_metadata
        prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", "domain_prompt.txt"
        )
        with open(prompt_path, "r") as file:
            prompt_template = file.read()

        # Prepare data
        summary_dict = summarize_dataframe(df).reset_index()
        summary_dict = summary_dict.astype(str).fillna("null").to_dict(orient="records")
        sample_row = df.sample(1).to_dict(orient="records")[0]

        # Pre-format JSON
        formatted_summary = json.dumps(summary_dict, indent=2)
        formatted_sample = json.dumps(sample_row, indent=2)

        # Format prompt
        prompt = prompt_template.format(
            arff_metadata=str(arff_metadata),
            summary_dict=formatted_summary,  # Pre-formatted
            sample_row=formatted_sample,  # Pre-formatted
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        
        domain_context = json.loads(response.choices[0].message.content)
        
        return AgentOutput(
            result=domain_context,
            metadata={"agent": "DomainAgent", "prompt": prompt, "status": "success"}
        )
    
    # def analyze(self, df: pd.DataFrame, arff_metadata: str = "") -> Tuple[dict, str]:
    #     # Load prompt
    #     prompt_path = os.path.join(
    #         os.path.dirname(__file__), "prompts", "domain_prompt.txt"
    #     )
    #     with open(prompt_path, "r") as file:
    #         prompt_template = file.read()

    #     # Prepare data
    #     summary_dict = summarize_dataframe(df).reset_index()
    #     summary_dict = summary_dict.astype(str).fillna("null").to_dict(orient="records")
    #     sample_row = df.sample(1).to_dict(orient="records")[0]

    #     # Pre-format JSON
    #     formatted_summary = json.dumps(summary_dict, indent=2)
    #     formatted_sample = json.dumps(sample_row, indent=2)

    #     # Format prompt
    #     prompt = prompt_template.format(
    #         arff_metadata=str(arff_metadata),
    #         summary_dict=formatted_summary,  # Pre-formatted
    #         sample_row=formatted_sample,  # Pre-formatted
    #     )

    #     response = self.client.chat.completions.create(
    #         model="gpt-4.1-mini",
    #         messages=[{"role": "user", "content": prompt}],
    #         response_format={"type": "json_object"},
    #     )
    #     return json.loads(response.choices[0].message.content), prompt
