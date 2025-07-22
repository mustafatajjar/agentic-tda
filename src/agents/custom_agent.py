from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from src.agents.planner_agent import PlannerAgent, Action
from src.agents.domain_agent import DomainAgent
from src.agents.augment import AugmentAgent
from src.agents.eval_agent import EvaluationAgent
from src.utils import arff_to_dataframe, extract_arff_metadata
from src.agents.feature_pruning import prune_features_binary_classification
from sklearn.model_selection import KFold


load_dotenv()  # Load API keys


def run_with_custom_agents(
    planner_prompt=None,
    domain_prompt=None,
    augment_prompt=None,
    verbose=False,
):
    arff_file_path = "./data/dataset_31_credit-g.arff"
    metadata = extract_arff_metadata(arff_file_path)
    df = arff_to_dataframe(arff_file_path)
    target_column = "class"

    planner_agent = PlannerAgent()
    domain_agent = DomainAgent()
    augment_agent = AugmentAgent()
    target_column = "class"
    evaluator = EvaluationAgent(df, label=target_column)

    # Test on holdout before any augmentation
    original_eval = evaluator.test_on_holdout(df)
    print("Original holdout evaluation:", original_eval)

    # Nested cross-validation before any augmentation
    original_nested_cv_scores = evaluator.nested_cross_val(df)
    print("Original nested CV scores:", np.mean(original_nested_cv_scores))
    evals = [np.mean(original_nested_cv_scores)]

    i = 0
    max_augmentations = 1

    df = arff_to_dataframe(arff_file_path)  # your DataFrame

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_indices = []

    for _, test_index in kf.split(df):
        fold_indices.append(list(test_index))

    # Save each fold's indices to a file
    for i_fold, indices in enumerate(fold_indices):
        with open(f"outputs/fold_{i_fold+1}_indices.txt", "w") as f:
            f.write("\n".join(map(str, indices)))

    while True:
        if i > max_augmentations:
            break
        # get context from Domain Agent
        context, prompt = domain_agent.analyze(df, arff_metadata=metadata)

        # Augmentation step
        augmented_df, aa_prompt, aa_response = augment_agent.add_column(
            df.copy(), domain_context=context
        )

        try:
            y = augmented_df[target_column]
            X = augmented_df.drop(columns=[target_column])
            selected_features = prune_features_binary_classification(
                X, y, time_limit_per_split=100, eval_metric="accuracy"
            )
            X_pruned = X[selected_features]
            augmented_df_pruned = X_pruned.copy()
            augmented_df_pruned[target_column] = y
        except Exception as e:
            print(f"Feature pruning failed: {e}")
            augmented_df_pruned = augmented_df

        # Nested cross-validation on the new table
        augmented_eval_scores = evaluator.nested_cross_val(augmented_df_pruned)
        print("Nested CV scores after augmentation:", augmented_eval_scores)
        evals.append(np.mean(augmented_eval_scores))

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
                file.write(str(np.mean(augmented_eval_scores)))

        # remove column and inform planner if eval dropped
        if augmented_eval_scores and np.mean(augmented_eval_scores) < max(evals):
            augmented_df = augmented_df.drop(
                columns=[augment_agent.latest_added_column]
            )
        planner_agent.last_improved = False
        if augmented_df is not None:
            df = augmented_df.copy()
        else:
            print("Augmentation failed, skipping this iteration.")
        i += 1

    # Test on holdout after all augmentations
    final_eval = evaluator.test_on_holdout(df)
    print("Final holdout evaluation:", final_eval)
    return final_eval
