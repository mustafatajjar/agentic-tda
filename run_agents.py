import pandas as pd
from agents.planner_agent import PlannerAgent, Action
from agents.domain_agent import DomainAgent
from agents.augment import AugmentAgent
from dotenv import load_dotenv

load_dotenv()  # Load API keys

def main():
    # Initialize agents
    domain_agent = DomainAgent()
    augment_agent = AugmentAgent()
    # also add eval_agent here
    
    # 1. get context of table + augment table 
    # Load data (replace with your data source), simple example need to convert here real table to df aswell
    df = pd.DataFrame({
        "City": ["Berlin", "Tokyo"],
        "Population": [3769000, 13960000]
    })
    context = domain_agent.analyze(df)
    print(f"\nDomain Context: {context}")
    augmented_df = augment_agent.augment_dataframe(context, df, num_suggestions=2)
    print(augmented_df)
    
    # 2. Use eval agent here for simple workflow


if __name__ == "__main__":
    main()