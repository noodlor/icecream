#!/usr/bin/env python3

import argparse
import sys
import math

def main():
    parser = argparse.ArgumentParser(
        description="Calculate optimal panel size OR detectable difference for a sensory panel.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Usage Examples:\n"
               "  Find Tasters Needed:    python3 panel_size.py -b 5 -d 1.0\n"
               "  Find Detectable Diff:   python3 panel_size.py -b 5 -t 15"
    )
    
    parser.add_argument('-b', '--brands', type=int, metavar='BRANDS', required=True,
                        help='Number of brands/samples you need to test')
    parser.add_argument('-d', '--difference', type=float, metavar='DIFF', default=1.0,
                        help='The minimum score difference you want to detect (default: 1.0 points)')
    parser.add_argument('-t', '--tasters', type=int, metavar='TASTERS',
                        help='Number of tasters you have (Triggers "Reverse Math" mode to find detectable diff)')
    parser.add_argument('-s', '--stdev', type=float, metavar='STD', default=1.3,
                        help='Estimated standard deviation of your panel (default: 1.3 for a 9-pt scale)')
    parser.add_argument('-u', '--unbalanced', action='store_true',
                        help='Skip serving order math and allow an unbalanced panel size')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    b = args.brands
    sigma = args.stdev
    z_constant = 7.84 # Represents (1.96 + 0.84)^2 for 95% Confidence and 80% Power

    if b < 2:
        print("\nERROR: You must be testing at least 2 brands to run a comparison.")
        sys.exit(1)

    # =========================================================
    # MODE 1: REVERSE MATH (User provided Tasters, solve for Difference)
    # =========================================================
    if args.tasters:
        n = args.tasters
        if n <= 0:
            print("\nERROR: Number of tasters must be a positive number.")
            sys.exit(1)
            
        # Algebra reversed: delta = sqrt( (2 * z_constant * sigma^2) / n )
        calculated_delta = math.sqrt((2 * z_constant * (sigma ** 2)) / n)
        remainder = n % b

        print("\n" + "="*55)
        print(" 📋 SENSORY PANEL DETECTABLE DIFFERENCE CALCULATOR")
        print("="*55)
        print(f"Brands Being Tested:      {b}")
        print(f"Available Tasters:        {n}")
        print("-" * 55)
        
        print("1️⃣ STATISTICAL POWER:")
        print(f"   With {n} tasters, your panel has 80% power to")
        print(f"   reliably detect a difference of {calculated_delta:.2f} points")
        print(f"   (or larger) on a 9-point scale. Anything smaller")
        print(f"   might be missed by the math.")
        print("")
        print("2️⃣ EXPERIMENTAL BALANCE:")
        if remainder == 0:
            print(f"   Excellent! {n} is a perfect multiple of {b}.")
            print(f"   Your serving schedules will be naturally balanced.")
        else:
            print(f"   Warning: {n} is not a multiple of {b}.")
            print(f"   You cannot create a perfectly balanced serving schedule.")
        print("="*55 + "\n")
        sys.exit(0)

    # =========================================================
    # MODE 2: ORIGINAL MATH (User provided Difference, solve for Tasters)
    # =========================================================
    delta = args.difference
    raw_n = 2 * z_constant * ((sigma / delta) ** 2)
    min_tasters = math.ceil(raw_n)
    remainder = min_tasters % b
    
    if args.unbalanced:
        optimal_tasters = min_tasters
    elif remainder == 0:
        optimal_tasters = min_tasters
    else:
        optimal_tasters = min_tasters + (b - remainder)

    print("\n" + "="*55)
    print(" 📋 SENSORY PANEL SIZE OPTIMIZER")
    print("="*55)
    print(f"Brands Being Tested:      {b}")
    print(f"Target Detectable Diff:   {delta} points (on 9-pt scale)")
    if args.unbalanced:
        print("Mode:                     UNBALANCED DESIGN ALLOWED")
    print("-" * 55)
    
    print("1️⃣ STATISTICAL THRESHOLD:")
    print(f"   To achieve 80% Statistical Power, you need an")
    print(f"   absolute minimum of {min_tasters} tasters. Any fewer,")
    print(f"   and your T-Tests will likely fail to detect winners.")
    print("")
    print("2️⃣ EXPERIMENTAL BALANCE:")
    
    if args.unbalanced:
        print(f"   You have chosen to bypass the balance multiplier.")
        print(f"   Because {min_tasters} is not a multiple of {b}, you cannot")
        print(f"   create a perfectly symmetrical serving schedule.")
        print(f"   Make sure to randomize serving order manually!")
    elif optimal_tasters == min_tasters:
        print(f"   Lucky you! Your minimum requirement ({min_tasters}) is already")
        print(f"   a perfect multiple of {b}, meaning your serving")
        print(f"   schedule will be naturally balanced.")
    else:
        print(f"   To completely eliminate serving-order bias, your")
        print(f"   panel size must be a multiple of {b}.")
        print(f"   (Rounding {min_tasters} up to {optimal_tasters})")
        
    print("-" * 55)
    print(f"🎯 RECOMMENDED PANEL SIZE: {optimal_tasters} TASTERS")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
