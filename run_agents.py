import pandas as pd
from agents.planner_agent import PlannerAgent, Action
from agents.domain_agent import DomainAgent
from agents.augment import AugmentAgent
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
    
    #3. get context of table + augment table 
    context = domain_agent.analyze(df, arff_metadata=metadata)
    print("Domain Context:")
    print(context)
    
    
    # TODO: need to finish augment properly + planner agent
    
    augmented_df = augment_agent.add_column(
        df,
        domain_context=context
    )
    print("Augmented DataFrame:")
    print(augmented_df.head())
    '''
    # 4. Use eval agent here for simple workflow
    '''
if __name__ == "__main__":
    main()