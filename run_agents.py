import pandas as pd
from agents.planner_agent import PlannerAgent, Action
from agents.domain_agent import DomainAgent
from agents.augment import AugmentAgent
from dotenv import load_dotenv
from scipy.io import arff
from agents.eval_agent import EvaluationAgent

load_dotenv()  # Load API keys

# maybe make util.py file for this kind of functions
def load_arff_to_dataframe(file_path):
    """
    Load ARFF file into pandas DataFrame
    
    Parameters:
    - file_path: path to the ARFF file
    
    Returns:
    - pandas DataFrame
    """
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Decode byte strings to regular strings if needed
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    
    for col in str_df:
        df[col] = str_df[col]
        
    return df

def main():
    #1.  load data set here
    df = load_arff_to_dataframe("data/dataset.arff")
    
    #2.  Initialize agents
    domain_agent = DomainAgent()
    augment_agent = AugmentAgent()
    #eval_agent = EvaluationAgent()
    
    #3. get context of table + augment table 
    context = domain_agent.analyze(df)
    augmented_df = augment_agent.augment_dataframe(
        domain_context=context,
        df=df,
        num_suggestions=3
    )
    
    '''
    # 3. Use eval agent here for simple workflow
    results = eval_agent.compare_tables(
        original_df=df,
        augmented_df=augmented_df,
        target_column="binaryClass"
    )

    
    # 4. Display results
    print("\nPerformance Comparison:")
    print(pd.DataFrame({
        'Metric': ['Accuracy', 'F1-Score'],
        'Original': [results['baseline']['accuracy'], results['baseline']['f1']],
        'Augmented': [results['augmented']['accuracy'], results['augmented']['f1']],
        'Improvement': [results['improvement']['accuracy'], results['improvement']['f1']]
    }))
    
    print("\nTop Feature Improvements:")
    print(results['feature_impact'].head())
    '''
if __name__ == "__main__":
    main()