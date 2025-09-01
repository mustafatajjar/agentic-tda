from enum import Enum, auto
import os
from openai import OpenAI
from src.agents.core.agent import Agent, AgentInput, AgentOutput
from dataclasses import dataclass
from typing import Any, Dict 
import pandas as pd

class Action(Enum):
    AUGMENT = auto()
    EVALUATE = auto()
    STOP = auto()

@dataclass
class PlannerAgentInput:
    df: pd.DataFrame
    context: Dict

class PlannerAgent(Agent):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.history = []
        self.last_improved = (
            True  # flag to store if the performance did not improve last step
        )
    
    def run(self, input: AgentInput):
        """Dynamic decision-making with LLM"""
        planner_input = PlannerAgentInput(input.data)
        context = planner_input.context
        df = planner_input.df
        
        last_aug_improved_notice = (
            "Last iteration, the augmentation did not lead to improvement!"
            if not self.last_improved
            else ""
        )
        prompt = f"""
        **Table State**: Columns={df.columns.tolist()}
        **Domain Context**: {context}
        **Action History**: {self.history[-3:] if self.history else "None"}
        {last_aug_improved_notice}
        
        Choose:
        - AUGMENT: If context suggests valuable additions
        - EVALUATE: If table is feature-complete
        - STOP: If no improvements possible
        
        Respond ONLY with: AUGMENT|EVALUATE|STOP
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        decision = response.choices[0].message.content.strip().upper()
        self.history.append(decision)
        return AgentOutput(result=Action[decision])
    
    # def decide_next_action(self, df, context: dict) -> Action:
    #     """Dynamic decision-making with LLM"""
    #     last_aug_improved_notice = (
    #         "Last iteration, the augmentation did not lead to improvement!"
    #         if not self.last_improved
    #         else ""
    #     )
    #     prompt = f"""
    #     **Table State**: Columns={df.columns.tolist()}
    #     **Domain Context**: {context}
    #     **Action History**: {self.history[-3:] if self.history else "None"}
    #     {last_aug_improved_notice}
        
    #     Choose:
    #     - AUGMENT: If context suggests valuable additions
    #     - EVALUATE: If table is feature-complete
    #     - STOP: If no improvements possible
        
    #     Respond ONLY with: AUGMENT|EVALUATE|STOP
    #     """
    #     response = self.client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages=[{"role": "user", "content": prompt}],
    #         temperature=0.1,
    #     )
    #     decision = response.choices[0].message.content.strip().upper()
    #     self.history.append(decision)
    #     return Action[decision]
