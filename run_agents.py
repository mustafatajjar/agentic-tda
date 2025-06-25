import pandas as pd
from agents.planner_agent import PlannerAgent, Action
from agents.domain_agent import DomainAgent
from agents.augment import AugmentAgent
from dotenv import load_dotenv

load_dotenv()  # Load API keys

def main():
    # Initialize agents
    planner = PlannerAgent()
    domain_agent = DomainAgent()
    augment_agent = AugmentAgent()
    
    # Load data (replace with your data source)
    df = pd.DataFrame({
        "City": ["Berlin", "Tokyo"],
        "Population": [3769000, 13960000]
    })
    context = domain_agent.analyze(df)
    print(f"\nDomain Context: {context}")
    new_columns = augment_agent.suggest_columns(context, df)
    print(f"Suggested Columns: {new_columns}")
    
    """
    # Agentic loop
    max_iterations = 1
    for _ in range(max_iterations):
        # Step 1: Domain analysis
        context = domain_agent.analyze(df)
        print(f"\nDomain Context: {context}")
        
        # Step 2: Planner decision
        action = planner.decide_next_action(df, context)
        print(f"Planner Decision: {action.name}")
        
        if action == Action.STOP:
            break
        elif action == Action.AUGMENT:
            # Step 3: Augmentation
            df, result = augment_agent.augment(df, context)
            print(f"Augmentation Result: {result}")
            print("Augmented Table:")
            print(df.head())
        elif action == Action.EVALUATE:
            # (Implement your evaluation logic)
            print("Evaluation skipped in this demo")

    # Final output
    print("\nFinal DataFrame:")
    print(df)
    """

if __name__ == "__main__":
    main()