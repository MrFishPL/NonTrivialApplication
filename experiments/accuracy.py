import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

import pandas as pd


def _decimals_count(d: Decimal) -> int:
    """How many decimal places the Decimal has (non-negative)."""
    exp = d.as_tuple().exponent
    return -exp if exp < 0 else 0


def _to_scaled_int(d: Decimal, scale_decimals: int) -> int:
    """
    Convert Decimal to scaled integer with rounding to 'scale_decimals' places.
    Example: scale_decimals=2 -> 12.345 -> 1235 (rounded half up).
    """
    q = Decimal(1).scaleb(-scale_decimals)  # 10^-scale_decimals
    d_q = d.quantize(q, rounding=ROUND_HALF_UP)
    scale = 10 ** scale_decimals
    return int((d_q * scale).to_integral_value(rounding=ROUND_HALF_UP))


def _parse_decimal(s: str) -> Decimal | None:
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        d = Decimal(s)
    except Exception:
        return None
    # Filter NaN / Infinity-like values if they appear
    if not d.is_finite():
        return None
    return d


def _convolve_counts(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
    """
    Convolution of sum-distributions in 'counts space' (exact big ints):
    if a[u] = number of ways to get sum u (ordered draws),
       b[v] = number of ways to draw value v in 1 step (ordered),
    then (a * b)[u+v] += a[u] * b[v].
    """
    if not a or not b:
        return {}
    out = defaultdict(int)
    for su, cu in a.items():
        for v, cv in b.items():
            out[su + v] += cu * cv
    return dict(out)


def _sum_distribution_for_k_draws(freq: dict[int, int], k: int) -> dict[int, int]:
    """Distribution of sums for k ordered draws with replacement, weighted by frequencies."""
    dist = {0: 1}
    for _ in range(k):
        dist = _convolve_counts(dist, freq)
        if not dist:
            break
    return dist


def main(
    data_path: str,
    x_str: str,
    target_str: str,
    opponents: int,
    filter_data_min: str | None,
    filter_data_max: str | None,
    max_scale_decimals: int,
) -> None:
    # Load numbers as strings to keep decimals exact-ish
    df = pd.read_csv(data_path, dtype=str)
    if "number" not in df.columns:
        raise ValueError("CSV must contain a 'number' column.")

    raw = df["number"].dropna().tolist()
    nums_dec: list[Decimal] = []
    for s in raw:
        d = _parse_decimal(s)
        if d is not None:
            nums_dec.append(d)

    if not nums_dec:
        raise ValueError("No valid numeric data found in column 'number'.")

    x_dec = _parse_decimal(x_str)
    if x_dec is None:
        raise ValueError("Invalid --x value.")

    target_dec = _parse_decimal(target_str)
    if target_dec is None:
        raise ValueError("Invalid --target value.")

    fmin_dec = _parse_decimal(filter_data_min) if filter_data_min is not None else None
    fmax_dec = _parse_decimal(filter_data_max) if filter_data_max is not None else None

    # Optional filtering
    if fmin_dec is not None:
        nums_dec = [d for d in nums_dec if d >= fmin_dec]
    if fmax_dec is not None:
        nums_dec = [d for d in nums_dec if d <= fmax_dec]

    if not nums_dec:
        raise ValueError("No valid data left after filtering.")

    # Determine scaling decimals
    needed = max(_decimals_count(d) for d in (nums_dec + [x_dec, target_dec]))
    scale_decimals = min(needed, max_scale_decimals)

    # Scale everything to ints
    nums_int = [_to_scaled_int(d, scale_decimals) for d in nums_dec]
    x = _to_scaled_int(x_dec, scale_decimals)
    target = _to_scaled_int(target_dec, scale_decimals)

    N = len(nums_int)
    if N == 0:
        raise ValueError("Empty dataset after processing.")

    # Frequency of values (each row is equally likely in a draw)
    freq_all = Counter(nums_int)

    # Split into values strictly less/greater than x
    freq_lt = {v: c for v, c in freq_all.items() if v < x}
    freq_gt = {v: c for v, c in freq_all.items() if v > x}

    # Compute distributions for sums of opponents draws
    dist_lt = _sum_distribution_for_k_draws(freq_lt, opponents)
    dist_gt = _sum_distribution_for_k_draws(freq_gt, opponents)

    # Threshold for s = sum(opponents): compare to (target - x)
    thr = target - x

    # Win counts (ordered tuples)
    # Case S > 0: need all opponents < x AND s < (target - x)
    win_pos = sum(c for s, c in dist_lt.items() if s < thr)

    # Case S < 0: need all opponents > x AND s > (target - x)
    win_neg = sum(c for s, c in dist_gt.items() if s > thr)

    win_total = win_pos + win_neg
    total_games = N ** opponents  # ordered draws with replacement

    frac = Fraction(win_total, total_games) if total_games > 0 else Fraction(0, 1)
    pct = float(frac) * 100.0

    # Pretty printing
    scale_info = f"10^{scale_decimals}" if scale_decimals > 0 else "1"
    x_back = (Decimal(x) / (Decimal(10) ** scale_decimals)) if scale_decimals else Decimal(x)
    thr_back = (Decimal(thr) / (Decimal(10) ** scale_decimals)) if scale_decimals else Decimal(thr)
    target_back = (Decimal(target) / (Decimal(10) ** scale_decimals)) if scale_decimals else Decimal(target_back)

    print("\n=== Exact win rate for fixed x (all combinations) ===")
    print(f"Data rows (N): {N:,}")
    print(f"Opponents draws: {opponents}  (with replacement, ordered)")
    print(f"Scaling used: {scale_info}  (numbers rounded to {scale_decimals} decimals before counting)")
    print(f"x: {x_back}")
    print(f"Target total: {Decimal(target) / (Decimal(10) ** scale_decimals) if scale_decimals else target}")
    print(f"Threshold (target - x): {thr_back}\n")

    print(f"Winning games (S>0 branch): {win_pos:,}")
    print(f"Winning games (S<0 branch): {win_neg:,}")
    print(f"Winning games (total):      {win_total:,}")
    print(f"All games (total):          {total_games:,}\n")

    print(f"Win probability (exact): {frac.numerator}/{frac.denominator}")
    print(f"Win percentage:          {pct:.10f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Exact Win Percentage",
        description=(
            "Compute exact P(win) for a given x by enumerating all ordered opponent draws with replacement "
            "from an empirical distribution (CSV column: number)."
        ),
    )

    parser.add_argument("-dp", "--data-path", type=str, required=True, help="Path to CSV with column 'number'.")
    parser.add_argument("--x", type=str, required=True, help="Your chosen number x (parsed as Decimal).")
    parser.add_argument("--target", type=str, default="50", help="Target total (default: 50).")
    parser.add_argument("--opponents", type=int, default=4, help="How many opponents are drawn (default: 4).")

    parser.add_argument("--filter-data-min", type=str, default=None,
                        help="Optional: drop data values below this threshold.")
    parser.add_argument("--filter-data-max", type=str, default=None,
                        help="Optional: drop data values above this threshold.")

    parser.add_argument("--max-scale-decimals", type=int, default=6,
                        help="Max decimals used for exact scaling. If data has more, it will be rounded.")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        x_str=args.x,
        target_str=args.target,
        opponents=args.opponents,
        filter_data_min=args.filter_data_min,
        filter_data_max=args.filter_data_max,
        max_scale_decimals=args.max_scale_decimals,
    )