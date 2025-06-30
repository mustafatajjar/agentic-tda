from enum import Enum, auto
import os
from openai import OpenAI


class Action(Enum):
    AUGMENT = auto()
    EVALUATE = auto()
    STOP = auto()


class PlannerAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.history = []

    def decide_next_action(self, df, context: dict) -> Action:
        """Dynamic decision-making with LLM"""
        prompt = f"""
        **Table State**: Columns={df.columns.tolist()}
        **Domain Context**: {context}
        **Action History**: {self.history[-3:] if self.history else "None"}
        
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
        return Action[decision]
