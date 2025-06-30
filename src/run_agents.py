import pandas as pd
from dotenv import load_dotenv

from agents.planner_agent import PlannerAgent, Action
from agents.domain_agent import DomainAgent
from agents.augment import AugmentAgent
from agents.eval_agent import evaluate
from utils import arff_to_dataframe, extract_arff_metadata

load_dotenv()  # Load API keys


def main():
    # 1.  load data set here
    arff_file_path = "./data/dataset_37_diabetes.arff"

    metadata = extract_arff_metadata(arff_file_path)
    df = arff_to_dataframe(arff_file_path)

    # 2.  Initialize agents
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
        augmented_df = augment_agent.add_column(df, domain_context=context)
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
