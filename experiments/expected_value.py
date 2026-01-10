import argparse
import pandas as pd
import numpy as np


def main(
    data_path: str,
    x: float,
    target: float,
    opponents: int,
    filter_data_min: float | None,
    filter_data_max: float | None,
) -> None:
    # Load data
    s = pd.read_csv(data_path)["number"]
    numbers = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    numbers = numbers[np.isfinite(numbers)]

    # Optional filtering
    if filter_data_min is not None:
        numbers = numbers[numbers >= filter_data_min]
    if filter_data_max is not None:
        numbers = numbers[numbers <= filter_data_max]

    if len(numbers) == 0:
        raise ValueError("No valid data left after loading / filtering.")

    m = float(numbers.mean())  # E[X]
    ey = opponents * m         # E[Y] where Y = sum of opponents' draws

    # Expected payoff for choosing x:
    # E[x * (target - x - Y)] = x * (target - x - E[Y])
    ev = x * (target - x - ey)

    print("\n=== Expected value for fixed x ===")
    print(f"Rows in data: {len(numbers):,}")
    print(f"E[X] (mean of data): {m:.10f}")
    print(f"Opponents: {opponents} -> E[Y] = {ey:.10f}")
    print(f"Target: {target}")
    print(f"x: {x}")
    print(f"\nE[payoff] = {ev:.10f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Expected Value",
        description="Compute expected payoff for a given x using empirical mean from CSV column 'number'.",
    )

    parser.add_argument("-dp", "--data-path", type=str, required=True, help="Path to CSV with column 'number'.")
    parser.add_argument("--x", type=float, required=True, help="Your chosen number x.")
    parser.add_argument("--target", type=float, default=50.0, help="Target total (default: 50).")
    parser.add_argument("--opponents", type=int, default=4, help="Number of opponents (default: 4).")

    parser.add_argument("--filter-data-min", type=float, default=None,
                        help="Optional: filter out data values below this threshold.")
    parser.add_argument("--filter-data-max", type=float, default=None,
                        help="Optional: filter out data values above this threshold.")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        x=args.x,
        target=args.target,
        opponents=args.opponents,
        filter_data_min=args.filter_data_min,
        filter_data_max=args.filter_data_max,
    )
