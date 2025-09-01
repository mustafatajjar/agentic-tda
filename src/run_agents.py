from argparse import ArgumentParser
from dotenv import load_dotenv


from src.agents.core.tda_agent import TDAAgent

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

    agent = TDAAgent(
        data_path=args.data_path,
        num_columns_to_add=args.num_columns_to_add,
        target_column="class",
        n_folds=10,
        test_size=0.2,
        model="tabpfn",
        verbose=verbose,
    )
    agent.run()


if __name__ == "__main__":
    main()
