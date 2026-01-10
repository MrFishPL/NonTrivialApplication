import argparse
import os
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

_GLOBAL_NUMBERS = None


def _init_worker(numbers: np.ndarray) -> None:
    """Initializer so we don't pickle the numbers array for every task."""
    global _GLOBAL_NUMBERS
    _GLOBAL_NUMBERS = numbers


def _simulate_chunk(n_sims: int, seed: int, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_sims draws of 4 opponents.
    For each draw, compute the open interval (low, high) of x values that would win.
    Return arrays of low and high endpoints for intervals that are non-empty after clipping to [x_min, x_max].
    """
    global _GLOBAL_NUMBERS
    rng = np.random.default_rng(seed)

    # Draw 4 opponents with replacement from the empirical distribution
    idx = rng.integers(0, len(_GLOBAL_NUMBERS), size=(n_sims, 4))
    opp = _GLOBAL_NUMBERS[idx]

    sum4 = opp.sum(axis=1)
    mx = opp.max(axis=1)
    mn = opp.min(axis=1)

    # T = 50 - (x2+x3+x4+x5)
    # Then S = T - x
    T = 50.0 - sum4

    # If S > 0 => T - x > 0 => x < T
    # In that regime the winner is the largest x, so you win iff x > max(opponents) and x < T
    # Winning interval: (mx, T)
    mask_pos = mx < T

    # If S < 0 => T - x < 0 => x > T
    # In that regime the winner is the smallest x, so you win iff x < min(opponents) and x > T
    # Winning interval: (T, mn)
    mask_neg = T < mn

    valid = mask_pos | mask_neg
    if not np.any(valid):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    low = np.empty(n_sims, dtype=np.float64)
    high = np.empty(n_sims, dtype=np.float64)

    if np.any(mask_pos):
        low[mask_pos] = mx[mask_pos]
        high[mask_pos] = T[mask_pos]

    if np.any(mask_neg):
        low[mask_neg] = T[mask_neg]
        high[mask_neg] = mn[mask_neg]

    low = low[valid]
    high = high[valid]

    # Clip intervals to the allowed search range for x
    low = np.maximum(low, x_min)
    high = np.minimum(high, x_max)

    # Keep only non-empty open intervals
    keep = high > low
    return low[keep], high[keep]


def _best_segment_from_intervals(starts: np.ndarray, ends: np.ndarray, x_min: float, x_max: float):
    """
    Given many open intervals (start, end), find a segment (a, b) where the number of covering intervals is maximal.
    Returns (best_segment, best_count).

    best_segment is (a, b) with a < b, and any x chosen strictly inside (a, b) achieves best_count overlaps.
    """
    if len(starts) == 0:
        return None, 0

    # Sweep-line events: start -> +1, end -> -1
    # For open intervals, we only choose x strictly inside, so any midpoint between event points is safe.
    vals = np.concatenate([starts, ends, np.array([x_min, x_max], dtype=np.float64)])
    deltas = np.concatenate([
        np.ones_like(starts, dtype=np.int64),
        -np.ones_like(ends, dtype=np.int64),
        np.array([0, 0], dtype=np.int64),
    ])

    # Tie-breaking at the same coordinate:
    # We want the region just after v to exclude intervals ending at v and include intervals starting at v.
    # So process ends (-1) BEFORE starts (+1) at the same v.
    # We'll sort by (value, kind) where kind: end=0, start=1, dummy=2
    kind = np.concatenate([
        np.ones_like(starts, dtype=np.int8),   # starts -> kind 1
        np.zeros_like(ends, dtype=np.int8),    # ends   -> kind 0
        np.array([2, 2], dtype=np.int8),       # dummy  -> kind 2
    ])

    order = np.lexsort((kind, vals))  # primary: vals, secondary: kind
    vals = vals[order]
    deltas = deltas[order]

    count = 0
    best_count = -1
    best_seg = None

    i = 0
    n = len(vals)
    while i < n:
        v = vals[i]

        # Apply all events at v (already ordered so ends first)
        while i < n and vals[i] == v:
            count += int(deltas[i])
            i += 1

        # The segment (v, next_v) has constant 'count'
        if i < n:
            next_v = vals[i]
            if next_v > v and count > best_count:
                best_count = count
                best_seg = (v, next_v)

    return best_seg, best_count


def main(data_path: str,
         n_sims: int,
         workers: int,
         chunk_size: int,
         seed: int,
         x_min: float,
         x_max: float,
         filter_data_min: float | None,
         filter_data_max: float | None) -> None:

    # Load data
    s = pd.read_csv(data_path)["number"]
    numbers = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    numbers = numbers[np.isfinite(numbers)]

    # Optional filtering (useful to drop extreme / troll values)
    if filter_data_min is not None:
        numbers = numbers[numbers >= filter_data_min]
    if filter_data_max is not None:
        numbers = numbers[numbers <= filter_data_max]

    if len(numbers) == 0:
        raise ValueError("No valid data left after loading / filtering.")

    if x_max <= x_min:
        raise ValueError("x_max must be > x_min.")

    workers = max(1, workers)
    chunk_size = max(10_000, chunk_size)

    n_chunks = math.ceil(n_sims / chunk_size)
    sizes = [chunk_size] * (n_chunks - 1) + [n_sims - chunk_size * (n_chunks - 1)]

    lows_list: list[np.ndarray] = []
    highs_list: list[np.ndarray] = []

    # Parallel Monte Carlo simulation
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(numbers,)) as ex:
        futures = []
        for i, sz in enumerate(sizes):
            futures.append(ex.submit(_simulate_chunk, int(sz), seed + i, x_min, x_max))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Simulations (chunks)"):
            low, high = fut.result()
            lows_list.append(low)
            highs_list.append(high)

    starts = np.concatenate(lows_list) if lows_list else np.empty(0, dtype=np.float64)
    ends = np.concatenate(highs_list) if highs_list else np.empty(0, dtype=np.float64)

    # Find best x as a midpoint of the best overlap segment
    best_seg, best_count = _best_segment_from_intervals(starts, ends, x_min, x_max)

    win_prob_est = best_count / float(n_sims)

    print("\n=== Result (maximize P[win]) ===")
    print(f"Simulations: {n_sims:,}")
    print(f"Workers (processes): {workers}")
    print(f"Search range for x: [{x_min}, {x_max}]")
    print(f"Fraction of simulations with any winning x in range: {len(starts)/n_sims:.4f}")

    if best_seg is None:
        print("No winning x found in the given range (in this simulation).")
        return

    a, b = best_seg
    x_best = (a + b) / 2.0  # Safe: strictly inside (a,b)
    print(f"\nBest x-interval: ({a:.6g}, {b:.6g})")
    print(f"Suggested x (midpoint): {x_best:.6f}")
    print(f"Estimated P(win) for this x: {win_prob_est:.6f}")

    # Optional: evaluate the best integer x in [ceil(x_min), floor(x_max)]
    lo_int = int(math.ceil(x_min))
    hi_int = int(math.floor(x_max))
    if lo_int <= hi_int and len(starts) > 0:
        ints = list(range(lo_int, hi_int + 1))
        best_i = None
        best_i_prob = -1.0

        for i in tqdm(ints, desc="Checking integer x values"):
            # Strict inequalities for open intervals: start < i < end
            wins_i = int(np.sum((starts < i) & (i < ends)))
            p_i_all = wins_i / float(n_sims)

            if p_i_all > best_i_prob:
                best_i_prob = p_i_all
                best_i = i

        print(f"\nBest integer x: {best_i}")
        print(f"Estimated P(win) for x={best_i}: {best_i_prob:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Max Win Probability",
        description="Maximize P(win) in the game x_i * S using an empirical distribution from a CSV file."
    )

    parser.add_argument("-dp", "--data-path", type=str, required=True)
    parser.add_argument("--n-sims", type=int, default=2_000_000, help="Number of Monte Carlo simulations.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Number of worker processes (CPU).")
    parser.add_argument("--chunk-size", type=int, default=200_000,
                        help="Simulations per multiprocessing chunk.")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed.")

    parser.add_argument("--x-min", type=float, default=-50, help="Minimum allowed x (your choice).")
    parser.add_argument("--x-max", type=float, default=50.0, help="Maximum allowed x (your choice).")

    parser.add_argument("--filter-data-min", type=float, default=None,
                        help="Optional: filter out data values below this threshold.")
    parser.add_argument("--filter-data-max", type=float, default=None,
                        help="Optional: filter out data values above this threshold.")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        n_sims=args.n_sims,
        workers=args.workers,
        chunk_size=args.chunk_size,
        seed=args.seed,
        x_min=args.x_min,
        x_max=args.x_max,
        filter_data_min=args.filter_data_min,
        filter_data_max=args.filter_data_max,
    )