#!/usr/bin/env python3

import argparse
import sys
from scipy.stats import binom

def main():
    parser = argparse.ArgumentParser(
        description="Calculate statistical significance for Sensory Discrimination Tests.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Usage Examples:\n  Triangle Analysis: python3 discrimination_test.py -m triangle -t 30 -c 15\n  Tetrad Planning:   python3 discrimination_test.py -m tetrad -t 45"
    )
    
    # NEW: The Method Flag
    parser.add_argument('-m', '--method', choices=['triangle', 'tetrad', 'duotrio'], default='triangle',
                        help='Which test are you running? (default: triangle)')
    
    parser.add_argument('-t', '--tasters', type=int, metavar='TOTAL', required=True,
                        help='Total number of tasters who participated (or will participate)')
    parser.add_argument('-c', '--correct', type=int, metavar='CORRECT', 
                        help='Number of tasters who correctly identified the sample(s)')
    parser.add_argument('-a', '--alpha', type=float, metavar='ALPHA', default=0.05,
                        help='Significance level (default: 0.05 / 95%% confidence)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    n = args.tasters
    alpha = args.alpha

    if n <= 0:
        print("\nERROR: Total tasters must be a positive number.")
        sys.exit(1)

    # ---------------------------------------------------------
    # SET THE MATH BASED ON THE METHOD
    # ---------------------------------------------------------
    if args.method in ['triangle', 'tetrad']:
        p_guess = 1/3
    elif args.method == 'duotrio':
        p_guess = 1/2 # Duo-Trio gives a reference, so it's a 50/50 guess

    min_correct = int(binom.ppf(1 - alpha, n, p_guess)) + 1
    method_title = args.method.upper()

    print("\n" + "="*50)

    # ---------------------------------------------------------
    # MODE 1: PLANNING MODE
    # ---------------------------------------------------------
    if args.correct is None:
        print(f" 📊 {method_title} TEST TARGET CALCULATOR")
        print("="*50)
        print(f"Planned Tasters:    {n}")
        print(f"Guess Probability:  {p_guess:.2f} (by random chance)")
        print(f"Significance Level: {alpha} (95% confidence)")
        print("-" * 50)
        print(f"🎯 TARGET: {min_correct} CORRECT GUESSES")
        print(f"If you run a {method_title} test with {n} people, at least {min_correct} of them")
        print("must guess correctly to prove a statistically significant difference.")
        print("="*50 + "\n")
        sys.exit(0)

    # ---------------------------------------------------------
    # MODE 2: ANALYSIS MODE 
    # ---------------------------------------------------------
    k = args.correct
    
    if k > n:
        print("ERROR: You cannot have more correct guesses than total tasters!")
        sys.exit(1)
    if k < 0:
        print("ERROR: Correct guesses must be a positive number.")
        sys.exit(1)

    p_value = binom.sf(k - 1, n, p_guess)

    print(f" 📊 {method_title} TEST ANALYSIS REPORT")
    print("="*50)
    print(f"Total Tasters:      {n}")
    print(f"Correct Guesses:    {k}")
    print(f"Required to Pass:   {min_correct} (at alpha = {alpha})")
    print(f"Calculated p-value: {p_value:.5f}")
    print("-" * 50)

    if p_value < alpha:
        print("🟢 CONCLUSION: STATISTICALLY SIGNIFICANT DIFFERENCE")
        print("The panel successfully detected a difference between the samples.")
        print("You can be confident this wasn't just blind luck.")
    else:
        print("🔴 CONCLUSION: NO SIGNIFICANT DIFFERENCE DETECTED")
        print("The tasters could not reliably tell the samples apart.")
        print("Any correct guesses are mathematically indistinguishable from random chance.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
