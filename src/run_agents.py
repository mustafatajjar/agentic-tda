import pandas as pd
from agents.planner_agent import PlannerAgent, Action
from agents.domain_agent import DomainAgent
from agents.augment import AugmentAgent
from agents.eval_agent import evaluate
from dotenv import load_dotenv
import arff

load_dotenv()  # Load API keys

# TODO: Move to utils.py
def arff_to_dataframe(file_path):
    """Convert ARFF to pandas DataFrame."""
    with open(file_path, 'r') as f:
        arff_data = arff.load(f)
    data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])
    return data

def extract_arff_metadata(file_path):
    """Extract metadata comments (lines starting with %) from an ARFF file."""
    comments = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith('%'):
                comments.append(line.strip('%').strip())
    return "\n".join(comments)

def main():
    #1.  load data set here
    arff_file_path = "./data/dataset_37_diabetes.arff"
    
    metadata = extract_arff_metadata(arff_file_path)
    df = arff_to_dataframe(arff_file_path)
    
    #2.  Initialize agents
    domain_agent = DomainAgent()
    augment_agent = AugmentAgent()

    # evaluate first dataset
    original_eval = evaluate(df)

    i = 0
    
    while True:
        if i > 2:
            break
        # get context from Domain Agent
        context = domain_agent.analyze(df, arff_metadata=metadata)
        print("Domain Context:")
        print(context)
        
        # TODO: need to finish augment properly + planner agent
        augmented_df = augment_agent.add_column(
            df,
            domain_context=context
        )
        print("Current DataFrame:")
        print(df.head())
        print("Augmented DataFrame:")
        print(augmented_df.head())

        # evaluate new table
        augmented_eval = evaluate(augmented_df)
        print("Original evaluation:", original_eval)
        print("Augmented evaluation:", augmented_eval)

        df = augmented_df
        i += 1

if __name__ == "__main__":
    main()