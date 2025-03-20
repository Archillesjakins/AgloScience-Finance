#!/usr/bin/env python3
"""
Main Script for Algorithmic Sciences Quantitative Finance Project

This script serves as an entry point to run all analysis components in sequence:
1. Basic data analysis
2. Quantitative analysis
3. Trading strategy simulation
"""

import os
import time
import argparse
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and measure its execution time."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    exit_code = os.system(f"python3 {script_path}")
    
    execution_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: {description}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Exit code: {exit_code}")
    print(f"{'='*80}\n")
    
    return exit_code == 0

def main():
    """Main function to run all analysis components."""
    parser = argparse.ArgumentParser(description="Algorithmic Sciences Quantitative Finance Project")
    
    parser.add_argument("--basic", action="store_true", help="Run basic data analysis only")
    parser.add_argument("--quant", action="store_true", help="Run quantitative analysis only")
    parser.add_argument("--trading", action="store_true", help="Run trading strategy only")
    parser.add_argument("--all", action="store_true", help="Run all components (default)")
    
    args = parser.parse_args()
    
    # Default behavior if no specific component is selected
    run_all = args.all or not (args.basic or args.quant or args.trading)
    
    # Make sure output directory exists
    Path("output").mkdir(parents=True, exist_ok=True)
    
    # Print banner
    print("\n" + "*"*80)
    print("*" + " "*78 + "*")
    print("*" + "  ALGORITHMIC SCIENCES - QUANTITATIVE FINANCE PROJECT  ".center(78) + "*")
    print("*" + " "*78 + "*")
    print("*"*80 + "\n")
    
    # Run the selected components
    if args.basic or run_all:
        basic_analysis_success = run_script("src/analyze_data.py", "Basic Data Analysis")
        if not basic_analysis_success and run_all:
            print("Basic analysis failed, stopping execution.")
            return
    
    if args.quant or run_all:
        quant_analysis_success = run_script("src/quantitative_analysis.py", "Quantitative Finance Analysis")
        if not quant_analysis_success and run_all:
            print("Quantitative analysis failed, stopping execution.")
            return
    
    if args.trading or run_all:
        trading_success = run_script("src/trading_strategy.py", "Trading Strategy Simulation")
        if not trading_success:
            print("Trading strategy simulation failed.")
            return
    
    print("\nAll selected analysis components completed successfully!")
    print("\nResults and visualizations can be found in the 'output' directory.")

if __name__ == "__main__":
    main() 