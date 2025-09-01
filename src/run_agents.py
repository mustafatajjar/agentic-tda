from argparse import ArgumentParser
from dotenv import load_dotenv

from src.agents.tda_pipeline import TDAPipeline, TDAPipelineConfig, SimplifiedTDAPipeline
from src import logger

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
    parser.add_argument(
        "--use_simplified",
        action="store_true",
        help="Use simplified pipeline instead of route-based pipeline",
    )
    args = parser.parse_args()

    # Create configuration
    config = TDAPipelineConfig(
        data_path=args.data_path,
        num_columns_to_add=args.num_columns_to_add,
        target_column="class",
        n_folds=10,
        test_size=0.2,
        model="lightgbm",
        max_augmentations=10,
        verbose=verbose,
    )

    # For now, use simplified pipeline as the route-based one has data flow issues
    if args.use_simplified:
        # Use simplified pipeline (maintains original logic but with better structure)
        pipeline = SimplifiedTDAPipeline(config)
        logger.info("Using simplified TDA pipeline")
    else:
        # Route-based pipeline has data flow issues - use simplified instead
        logger.warning("Route-based pipeline has data flow issues. Using simplified pipeline instead.")
        pipeline = SimplifiedTDAPipeline(config)
        logger.info("Using simplified TDA pipeline (route-based disabled)")

    # Run the pipeline
    results = pipeline.run()
    
    # Print summary
    print(f"\n=== TDA Pipeline Results ===")
    print(f"Total iterations: {results.get('total_iterations', 'N/A')}")
    if 'final_score' in results:
        print(f"Final score: {results['final_score']:.4f}")
        print(f"Baseline score: {results['baseline_score']:.4f}")
        print(f"Improvement: {results['final_score'] - results['baseline_score']:.4f}")
    print(f"Augment responses: {len(results.get('augment_responses', []))}")
    print(f"Feature selection history: {len(results.get('selected_features_history', []))}")


if __name__ == "__main__":
    main()
