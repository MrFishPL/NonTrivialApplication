import argparse
import pandas as pd


def main(data_path: str, seed: int | None) -> None:
    # Load CSV
    df = pd.read_csv(data_path)

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Overwrite the same file
    df.to_csv(data_path, index=False)

    print(f"File shuffled and saved in-place: {data_path}")
    if seed is not None:
        print(f"Seed used: {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Shuffle CSV in place",
        description="Randomly shuffle rows in a CSV file and overwrite the same file.",
    )

    parser.add_argument(
        "-dp", "--data-path",
        type=str,
        required=True,
        help="Path to CSV file (will be overwritten)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible shuffling."
    )

    args = parser.parse_args()
    main(args.data_path, args.seed)