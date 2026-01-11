# Numerical Game (Non-trivial Fellowship application)

This repo contains the analysis code and experimental scripts we used to answer the **“choose a number” numerical game** question from the **Non-trivial Fellowship** application.

The key idea is **induction over human behavior**: instead of assuming perfectly rational opponents (Nash) or inventing a parametric distribution (e.g., Normal/Uniform), we approximate how *actual applicants* (humans) answer by fitting an **empirical distribution** from survey data, then optimize either:

1) **Probability of winning** (rank-based objective), or  
2) **Expected score** (payoff-based objective)

A short write-up of the approach, assumptions, and results is included as **`micropaper.pdf`**.

---

## The game

Five players submit real numbers: `x1, x2, x3, x4, x5`

We define:

- `S = 50 - (x1 + x2 + x3 + x4 + x5)`
- Your score: `u1 = x1 * S` (and similarly for other players)

Important ranking observation:

- If `S > 0`, everyone is multiplied by the same positive factor ⇒ **largest submitted number wins**
- If `S < 0`, everyone is multiplied by the same negative factor ⇒ **smallest submitted number wins**
- If `S = 0`, everyone ties

---

## Repository layout

Typical structure:

- `experiments/accuracy.py` — **exact win probability** for a fixed `x` (enumerates all ordered opponent draws from the empirical distribution)
- `experiments/expected_value.py` — expected payoff `E[x * (50 - x - Y)]` for a fixed `x`
- `experiments/max_E.py` — closed-form maximizer of expected value (`x = 25 - 2 * mean(data)`)
- `experiments/max_prob.py` — Monte Carlo optimizer for **max P(win)** using “winning intervals”
- `experiments/shuffle.py` — shuffle a CSV in-place
- `experiments/split.py` — shuffle + split into `train.csv` / `valid.csv`

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
