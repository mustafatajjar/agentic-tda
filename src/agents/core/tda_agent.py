from collections import Counter
from itertools import chain
import numpy as np
import os

from src.agents.planner_agent import PlannerAgent, Action
from src.agents.domain_agent import DomainAgent
from src.agents.augment import AugmentAgent
from src.agents.eval_agent import EvaluationAgent
from src.utils.funcs import arff_to_dataframe, extract_arff_metadata, write_to_logs
from src.utils.funcs import prune_features_binary_classification
from sklearn.model_selection import KFold
from src import logger

class TDAAgent:
    def __init__(
        self,
        data_path,
        num_columns_to_add=20,
        target_column="class",
        n_folds=10,
        test_size=0.2,
        model="tabpfn",
        verbose=True,
    ):
        self.data_path = data_path
        self.num_columns_to_add = num_columns_to_add
        self.target_column = target_column
        self.n_folds = n_folds
        self.test_size = test_size
        self.model = model
        self.verbose = verbose

        self.metadata = extract_arff_metadata(self.data_path)
        self.df = arff_to_dataframe(self.data_path)
        self.planner_agent = PlannerAgent()
        self.domain_agent = DomainAgent()
        self.augment_agent = AugmentAgent()
        self.evaluator = EvaluationAgent(
            self.df,
            label=self.target_column,
            n_folds=self.n_folds,
            test_size=self.test_size,
            model="lightgbm"
        )
        self.df_all_augmented = self.df.copy()
        self.augment_responses = []
        self.selected_features_history = []
        self.max_augmentations = 10

    def run(self):
        logger.debug(f"Loading ARFF file from: {self.data_path}")

        # Test on holdout before any augmentation
        original_eval = np.mean(
            self.evaluator.test_on_holdout_kfold(
                self.df, n_splits=self.n_folds, device="cuda"
            )
        )
        logger.debug(f"Original holdout evaluation: {original_eval}")

        # Nested cross-validation before any augmentation
        original_nested_cv_scores = self.evaluator.nested_cross_val(self.df)
        logger.debug(f"Original nested CV scores: {np.mean(original_nested_cv_scores)}")
        evals = [np.mean(original_nested_cv_scores)]

        df = arff_to_dataframe(self.data_path)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_indices = [list(test_index) for _, test_index in kf.split(df)]

        # Save each fold's indices to a file
        os.makedirs("outputs", exist_ok=True)
        for i, indices in enumerate(fold_indices):
            with open(f"outputs/fold_{i+1}_indices.txt", "w") as f:
                f.write("\n".join(map(str, indices)))

        j = 0
        while True:
            if j > self.max_augmentations:
                break
            # get context from Domain Agent
            context, prompt = self.domain_agent.analyze(df, arff_metadata=self.metadata)

            augmented_df, aa_prompt, aa_response, success = (
                self.augment_agent.add_column(
                    df.copy(),
                    domain_context=context,
                    history_responses=self.augment_responses,
                    selected_features_history=self.selected_features_history,
                    num_columns_to_add=self.num_columns_to_add,
                    target_column=self.target_column,
                )
            )
            if not success:
                logger.debug("Augmentation failed, stopping.")
                j += 1
                continue

            # Add any new columns from augmented_df to df_all_augmented,
            # only checking for duplicates with the original df columns
            for col in augmented_df.columns:
                if col not in self.df.columns:  # Only skip if in original df
                    self.df_all_augmented[col] = augmented_df[col]

            # Keep history of augment agent responses
            self.augment_responses.append({"response": aa_response})

            # Evaluate before pruning
            logger.debug("Evaluating augmented table before pruning...")
            eval_before_pruning_scores = self.evaluator.nested_cross_val(augmented_df)
            logger.debug(
                f"Nested CV scores on augmented table (before pruning): {np.mean(eval_before_pruning_scores)}"
            )

            try:
                y = augmented_df[self.target_column]
                X = augmented_df.drop(columns=[self.target_column])

                selected_features_per_split = prune_features_binary_classification(
                    X,
                    y,
                )

                # Combine features selected across splits (majority vote)
                feature_counts = Counter(
                    chain.from_iterable(selected_features_per_split)
                )
                num_splits = len(selected_features_per_split)
                selected_features = [
                    f
                    for f, count in feature_counts.items()
                    if count >= (num_splits // 2 + 1)
                ]

                self.selected_features_history.append(selected_features)

                logger.debug("Selected features:", selected_features)
                logger.debug("X columns:", X.columns.tolist())
                if isinstance(selected_features, str):
                    selected_features = [selected_features]
                if not all(f in X.columns for f in selected_features):
                    logger.debug(
                        "WARNING: Some selected features are not in DataFrame columns!"
                    )

                if len(selected_features) < len(X.columns) and all(
                    f in X.columns for f in selected_features
                ):
                    logger.debug(
                        f"Pruning effective: {len(X.columns)} -> {len(selected_features)} features."
                    )
                    X_pruned = X[selected_features]
                    augmented_df_pruned = X_pruned.copy()
                    augmented_df_pruned[self.target_column] = y
                else:
                    logger.debug(
                        "Pruning did not remove any features, using original augmented dataframe."
                    )
                    augmented_df_pruned = augmented_df
            except Exception as e:
                logger.debug(f"Feature pruning failed: {e}")
                augmented_df_pruned = augmented_df
                self.selected_features_history.append([])  # Keep empty if failed

            # Nested cross-validation on the new table
            eval_after_pruning_scores = self.evaluator.nested_cross_val(
                augmented_df_pruned,
            )
            evals.append(np.mean(eval_after_pruning_scores))
            logger.debug(
                f"Mean evaluation score after pruning: {np.mean(eval_after_pruning_scores)}"
            )

            # output file with prompt, response and eval
            if self.verbose:
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
                logger.debug(
                    "Performance improved. Keeping augmented and pruned table for next iteration."
                )
                df = augmented_df_pruned.copy()
                self.planner_agent.last_improved = True
            else:
                logger.debug(
                    "Performance did not improve. Reverting to table from previous iteration."
                )
                self.planner_agent.last_improved = False

            if np.mean(eval_after_pruning_scores) > 0.9999999:
                logger.debug("Perfect score achieved, stopping augmentation.")
                break

            j += 1

        # Test on holdout after all augmentations
        final_eval = np.mean(
            self.evaluator.test_on_holdout_kfold(
                df,
                n_splits=self.n_folds,
                device="cuda",
            )
        )

        logger.debug("Final holdout evaluation:", final_eval)
        logger.debug("original_eval:", original_eval)

        # Save the DataFrame with all columns ever added (including duplicates, except for original columns)
        self.df_all_augmented.to_csv("all_augmented_columns.csv", index=False)
        logger.debug("Saved all augmented columns to all_augmented_columns.csv")
