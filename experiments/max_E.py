import pandas as pd
import argparse


def main(data_path: str):
    numbers = pd.read_csv(data_path)["number"]
    numbers = numbers.to_numpy()
    
    m = numbers.mean()
    
    # Expected value logic:
    # Assume each opponent’s choice is a draw from the same distribution X.
    # Let Y = X2 + X3 + X4 + X5.
    # By linearity of expectation: E[Y] = E[X2] + ... + E[X5] = 4 * E[X].
    # Expected payoff for choosing x is:
    # E[x * (50 - x - Y)] = -x^2 + (50 - E[Y]) * x,
    # which is maximized at x = (50 - E[Y]) / 2.
    x = 25 - 2*m
    
    x = x.round(2)
    
    print(f"x for the max expected value: {x}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Max Expected Value',
        description='...',
    )
    
    parser.add_argument('-dp', '--data-path', type=str, required=True) 
    args = parser.parse_args()
    
    main(args.data_path)