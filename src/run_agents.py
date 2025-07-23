from argparse import ArgumentParser
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain

from src.agents.planner_agent import PlannerAgent, Action
from src.agents.domain_agent import DomainAgent
from src.agents.augment import AugmentAgent
from src.agents.eval_agent import EvaluationAgent
from src.utils import arff_to_dataframe, extract_arff_metadata, write_to_logs
from src.agents.feature_pruning import prune_features_binary_classification
from sklearn.model_selection import KFold

load_dotenv()  # Load API keys


def main(verbose=True):
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/dataset_31_credit-g.arff")
    parser.add_argument("--num_columns_to_add", type=int, default=10, help="Number of columns to add during augmentation")  # <-- Added argument
    args = parser.parse_args()

    # 1.  load data set here
    arff_file_path = args.data_path
    print(f"Loading ARFF file from: {arff_file_path}")

    metadata = extract_arff_metadata(arff_file_path)
    df = arff_to_dataframe(arff_file_path)

    # 2.  Initialize agents
    planner_agent = PlannerAgent()
    domain_agent = DomainAgent()
    augment_agent = AugmentAgent()
    target_column = "class"
    evaluator = EvaluationAgent(df, label=target_column, n_folds=10, test_size=0.2)

    # Test on holdout before any augmentation
    original_eval = evaluator.test_on_holdout(df)
    print("Original holdout evaluation:", original_eval)

    # Nested cross-validation before any augmentation
    original_nested_cv_scores = evaluator.nested_cross_val(df)
    print("Original nested CV scores:", np.mean(original_nested_cv_scores))
    evals = [np.mean(original_nested_cv_scores)]

    i = 0
    j = 0
    max_augmentations = 3

    df = arff_to_dataframe(arff_file_path)  # your DataFrame

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_indices = []

    for _, test_index in kf.split(df):
        fold_indices.append(list(test_index))

    # Save each fold's indices to a file
    for i, indices in enumerate(fold_indices):
        with open(f"outputs/fold_{i+1}_indices.txt", "w") as f:
            f.write("\n".join(map(str, indices)))

    while True:
        if j > max_augmentations:
            break
        # get context from Domain Agent
        context, prompt = domain_agent.analyze(df, arff_metadata=metadata)

        # TODO: need to finish augment properly + planner agent
        # augmented_df = augment_agent.mapping_binning_augment(df.copy(), domain_context=context)
        augmented_df, aa_prompt, aa_response = augment_agent.add_column(
            df.copy(), domain_context=context, num_columns_to_add=args.num_columns_to_add  # <-- Pass argument here
        )

        # Evaluate before pruning
        print("Evaluating augmented table before pruning...")
        eval_before_pruning_scores = evaluator.nested_cross_val(augmented_df)
        print(
            f"Nested CV scores on augmented table (before pruning): {np.mean(eval_before_pruning_scores)}"
        )

        try:
            y = augmented_df[target_column]
            X = augmented_df.drop(columns=[target_column])
            
            selected_features_per_split = prune_features_binary_classification(
                X, y, time_limit_per_split=200, eval_metric="accuracy"
            )
            
            # Combine features selected across splits (majority vote)
            feature_counts = Counter(chain.from_iterable(selected_features_per_split))
            num_splits = len(selected_features_per_split)
            selected_features = [f for f, count in feature_counts.items() if count >= (num_splits // 2 + 1)]

            if len(selected_features) < len(X.columns):
                print(f"Pruning effective: {len(X.columns)} -> {len(selected_features)} features.")
                X_pruned = X[selected_features]
                augmented_df_pruned = X_pruned.copy()
                augmented_df_pruned[target_column] = y
            else:
                print("Pruning did not remove any features, using original augmented dataframe.")
                augmented_df_pruned = augmented_df

        except Exception as e:
            print(f"Feature pruning failed: {e}")
            augmented_df_pruned = augmented_df

        # Nested cross-validation on the new table
        eval_after_pruning_scores = evaluator.nested_cross_val(augmented_df_pruned)
        evals.append(np.mean(eval_after_pruning_scores))
        print(
            f"Mean evaluation score after pruning: {np.mean(eval_after_pruning_scores)}"
        )

        # output file with prompt, response and eval
        if verbose:
            write_to_logs(
                prompt,
                context,
                aa_prompt,
                aa_response,
                original_eval,
                eval_before_pruning_scores,
                eval_after_pruning_scores,
            )

        # Decide whether to keep the changes based on performance
        if np.mean(eval_after_pruning_scores) > max(
            evals[:-1]
        ):  # Compare to previous best
            print(
                "Performance improved. Keeping augmented and pruned table for next iteration."
            )
            df = augmented_df_pruned.copy()
            planner_agent.last_improved = True
        else:
            print(
                "Performance did not improve. Reverting to table from previous iteration."
            )
            planner_agent.last_improved = False

        if np.mean(eval_after_pruning_scores) > 0.9999999:
            print("Perfect score achieved, stopping augmentation.")
            break

        j += 1

    # Test on holdout after all augmentations
    final_eval = evaluator.test_on_holdout(df, time_limit=600)
    print("Final holdout evaluation:", final_eval)


if __name__ == "__main__":
    main()
