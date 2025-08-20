from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from itertools import chain
import numpy as np
import os
import pandas as pd

from src.agents.planner_agent import PlannerAgent, Action
from src.agents.domain_agent import DomainAgent
from src.agents.augment import AugmentAgent
from src.agents.eval_agent import EvaluationAgent
from src.utils import arff_to_dataframe, extract_arff_metadata, write_to_logs
from src.agents.feature_pruning import prune_features_binary_classification,prune_features_sfs
from sklearn.model_selection import KFold

load_dotenv()  # Load API keys


def main(verbose=True):
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="./data/dataset_31_credit-g.arff"
    )
    parser.add_argument(
        "--num_columns_to_add",
        type=int,
        default=20,
        help="Number of columns to add during augmentation",
    )
    args = parser.parse_args()

    # Load data set
    arff_file_path = args.data_path
    print(f"Loading ARFF file from: {arff_file_path}")

    metadata = extract_arff_metadata(arff_file_path)
    df = arff_to_dataframe(arff_file_path)

    # Initialize agents
    planner_agent = PlannerAgent()
    domain_agent = DomainAgent()
    augment_agent = AugmentAgent()
    evaluator = EvaluationAgent(df, label=target_column, n_folds=10, test_size=0.2)

    target_column = 'class'
    model = "tabpfn"  # Specify the model to use for evaluation

    # Test on holdout before any augmentation
    # original_eval = evaluator.test_on_holdout(df,30*60)  # 30 minutes time limit
    original_eval = np.mean(evaluator.test_on_holdout_kfold_tabpfn(df, n_splits=10, device="cuda",method=model))
    print("Original holdout evaluation:", original_eval)

    # Nested cross-validation before any augmentation
    original_nested_cv_scores = evaluator.nested_cross_val(df,method=model)
    print("Original nested CV scores:", np.mean(original_nested_cv_scores))
    evals = [np.mean(original_nested_cv_scores)]

    i = 0
    j = 0
    max_augmentations = 10

    df = arff_to_dataframe(arff_file_path)  # your DataFrame

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_indices = []

    for _, test_index in kf.split(df):
        fold_indices.append(list(test_index))

    # Save each fold's indices to a file
    os.makedirs("outputs", exist_ok=True)
    for i, indices in enumerate(fold_indices):
        with open(f"outputs/fold_{i+1}_indices.txt", "w") as f:
            f.write("\n".join(map(str, indices)))

    augment_responses = []
    selected_features_history = []

    # Keep a DataFrame with all columns ever added (including duplicates, except for original columns)
    df_all_augmented = df.copy()

    while True:
        if j > max_augmentations:
            break
        # get context from Domain Agent
        context, prompt = domain_agent.analyze(df, arff_metadata=metadata)

        # TODO: need to finish augment properly + planner agent
        # augmented_df = augment_agent.mapping_binning_augment(df.copy(), domain_context=context)
        augmented_df, aa_prompt, aa_response, success = augment_agent.add_column(
            df.copy(),
            domain_context=context,
            history_responses=augment_responses,
            selected_features_history=selected_features_history,
            num_columns_to_add=args.num_columns_to_add,
            target_column=target_column,
        )
        if not success:
            print("Augmentation failed, stopping.")
            j += 1
            continue

        # Add any new columns from augmented_df to df_all_augmented,
        # only checking for duplicates with the original df columns
        for col in augmented_df.columns:
            if col not in df.columns:  # Only skip if in original df
                df_all_augmented[col] = augmented_df[col]

        # Keep history of augment agent responses
        augment_responses.append({"response": aa_response})

        # Evaluate before pruning
        print("Evaluating augmented table before pruning...")
        eval_before_pruning_scores = evaluator.nested_cross_val(augmented_df,method=model)
        print(
            f"Nested CV scores on augmented table (before pruning): {np.mean(eval_before_pruning_scores)}"
        )

        try:
            y = augmented_df[target_column]
            X = augmented_df.drop(columns=[target_column])

            selected_features_per_split = prune_features_binary_classification(
                X, y,  # <-- Use the specified model
            )

            
            # Combine features selected across splits (majority vote)
            feature_counts = Counter(chain.from_iterable(selected_features_per_split))
            num_splits = len(selected_features_per_split)
            selected_features = [
                f
                for f, count in feature_counts.items()
                if count >= (num_splits // 2 + 1)
            ]

            # selected_features_per_split = prune_features_sfs(
            #     X, y,  # <-- Use the specified model
            # )
            # selected_features = selected_features_per_split  # Use the first split's features for simplicity


            selected_features_history.append(selected_features)

            # Debug: print and fix selected_features if needed
            print("Selected features:", selected_features)
            print("X columns:", X.columns.tolist())
            if isinstance(selected_features, str):
                selected_features = [selected_features]
            if not all(f in X.columns for f in selected_features):
                print("WARNING: Some selected features are not in DataFrame columns!")

            if len(selected_features) < len(X.columns) and all(f in X.columns for f in selected_features):
                print(
                    f"Pruning effective: {len(X.columns)} -> {len(selected_features)} features."
                )
                X_pruned = X[selected_features]
                augmented_df_pruned = X_pruned.copy()
                augmented_df_pruned[target_column] = y
            else:
                print(
                    "Pruning did not remove any features, using original augmented dataframe."
                )
                augmented_df_pruned = augmented_df
        except Exception as e:
            print(f"Feature pruning failed: {e}")
            augmented_df_pruned = augmented_df
            selected_features_history.append([])  # Keep empty if failed

        # Nested cross-validation on the new table
        eval_after_pruning_scores = evaluator.nested_cross_val(augmented_df_pruned,method=model)
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
    # final_eval = evaluator.test_on_holdout(df, time_limit=30*60)
    final_eval = np.mean(evaluator.test_on_holdout_kfold_tabpfn(df, n_splits=10, device="cuda",method=model))

    print("Final holdout evaluation:", final_eval)
    print('original_eval:', original_eval)

    # Save the DataFrame with all columns ever added (including duplicates, except for original columns)
    df_all_augmented.to_csv("all_augmented_columns.csv", index=False)
    print("Saved all augmented columns to all_augmented_columns.csv")

if __name__ == "__main__":
    main()
