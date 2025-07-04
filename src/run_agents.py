from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

from src.agents.planner_agent import PlannerAgent, Action
from src.agents.domain_agent import DomainAgent
from src.agents.augment import AugmentAgent
from src.agents.eval_agent import evaluate
from src.utils import arff_to_dataframe, extract_arff_metadata

load_dotenv()  # Load API keys


def main(verbose=True):
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
        if i > 1:
            break
        # get context from Domain Agent
        context, prompt = domain_agent.analyze(df, arff_metadata=metadata)
        print("Domain Context:")
        print(context)

        # TODO: need to finish augment properly + planner agent
        # augmented_df = augment_agent.mapping_binning_augment(df.copy(), domain_context=context)
        augmented_df, aa_prompt, aa_response = augment_agent.add_column(df.copy(), domain_context=context)
        print("Current DataFrame:")
        print(df.head())
        print("Augmented DataFrame:")
        print(augmented_df.head())

        # evaluate new table
        augmented_eval = evaluate(augmented_df)
        print("Original evaluation:", original_eval)
        print("Augmented evaluation:", augmented_eval)

        # output file with prompt, response and eval
        if verbose:
            filename = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"outputs/run_{filename}.txt", "w") as file:
                file.write("DA Prompt:\n")
                file.write(prompt)
                file.write("\n\n")
                file.write("DA Response:\n")
                file.write(str(context))
                file.write("\n\n")

                file.write("\n" * 4)

                file.write("EA Prompt:\n")
                file.write(aa_prompt)
                file.write("\n\n")
                file.write("EA Response:\n")
                file.write(str(aa_response))
                file.write("\n\n")

                file.write("\n" * 4)

                file.write("Evaluation before augmenation:\n")
                file.write(str(original_eval))
                file.write("\n\n")
                file.write("Evaluation after augmenation:\n")
                file.write(str(augmented_eval))


        df = augmented_df.copy()
        i += 1


if __name__ == "__main__":
    main()
