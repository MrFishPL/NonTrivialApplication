import argparse
from pathlib import Path

import pandas as pd


def main(data_path: str, out_dir: str | None, valid_frac: float, seed: int | None) -> None:
    data_path_p = Path(data_path)

    if not (0.0 < valid_frac < 1.0):
        raise ValueError("--valid-frac must be between 0 and 1 (exclusive).")

    # Output directory
    if out_dir is None:
        out_dir_p = data_path_p.parent
    else:
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

    train_path = out_dir_p / "train.csv"
    valid_path = out_dir_p / "valid.csv"

    # Load CSV
    df = pd.read_csv(data_path_p)

    if len(df) == 0:
        raise ValueError("Input CSV is empty.")

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split
    n_valid = int(round(len(df) * valid_frac))
    n_valid = max(1, n_valid) if len(df) > 1 else 0  # sensownie dla bardzo małych plików
    df_valid = df.iloc[:n_valid]
    df_train = df.iloc[n_valid:]

    # Save
    df_train.to_csv(train_path, index=False)
    df_valid.to_csv(valid_path, index=False)

    print(f"Input: {data_path_p} (rows: {len(df)})")
    print(f"Train: {train_path} (rows: {len(df_train)})")
    print(f"Valid: {valid_path} (rows: {len(df_valid)})")
    if seed is not None:
        print(f"Seed used: {seed}")
    print(f"Valid fraction: {valid_frac}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Split CSV into train/valid",
        description="Randomly shuffle rows in a CSV file and split into train.csv and valid.csv.",
    )

    parser.add_argument(
        "-dp", "--data-path",
        type=str,
        required=True,
        help="Path to input CSV file (will NOT be overwritten).",
    )
    parser.add_argument(
        "-o", "--out-dir",
        type=str,
        default=None,
        help="Output directory for train.csv and valid.csv (default: same dir as input).",
    )
    parser.add_argument(
        "--valid-frac",
        type=float,
        default=0.1,
        help="Fraction of rows to put into valid split (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible shuffling.",
    )

    args = parser.parse_args()
    main(args.data_path, args.out_dir, args.valid_frac, args.seed)