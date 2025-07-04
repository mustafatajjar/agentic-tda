import pandas as pd
from dotenv import load_dotenv

from src.agents.planner_agent import PlannerAgent, Action
from src.agents.domain_agent import DomainAgent
from src.agents.augment import AugmentAgent
from src.agents.eval_agent import evaluate
from src.utils import arff_to_dataframe, extract_arff_metadata

load_dotenv()  # Load API keys


def main():
    # 1.  load data set here
    arff_file_path = "./data/dataset_31_credit-g.arff"

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
        #augmented_df = augment_agent.mapping_binning_augment(df.copy(), domain_context=context)
        augmented_df = augment_agent.add_column(df.copy(), domain_context=context)
        print("Current DataFrame:")
        print(df.head())
        print("Augmented DataFrame:")
        print(augmented_df.head())

        # evaluate new table
        augmented_eval = evaluate(augmented_df)
        print("Original evaluation:", original_eval)
        print("Augmented evaluation:", augmented_eval)

        df = augmented_df.copy()
        i += 1


if __name__ == "__main__":
    main()
