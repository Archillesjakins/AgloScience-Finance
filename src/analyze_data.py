#!/usr/bin/env python3
"""
Data Analysis Script for 200k.txt

This script analyzes the structure and content of the 200k.txt file
to understand the data distribution and patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

def load_data(file_path='200k.txt'):
    """Load the data from the CSV file."""
    print(f"Loading data from {file_path}...")
    start_time = time.time()
    
    # Read the data - semicolon separated values with no header
    # Based on the first few lines, we know the data has 8 columns
    df = pd.read_csv(file_path, header=None, sep=';')
    
    # Assign column names based on their position
    df.columns = [f'col_{i}' for i in range(len(df.columns))]
    
    # Report on the loading time
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")
    print(f"Dataset shape: {df.shape}")
    
    return df

def analyze_data(df):
    """Perform basic analysis on the dataset."""
    print("\n===== Data Analysis =====")
    
    # Basic statistics for each column
    print("\nBasic statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Check unique values in each column
    print("\nUnique values per column:")
    for col in df.columns:
        unique_values = df[col].nunique()
        print(f"{col}: {unique_values} unique values")
    
    # Sample of the data
    print("\nSample of the data (first 5 rows):")
    print(df.head())
    
    return df

def visualize_data(df, output_dir='./output'):
    """Create visualizations for the dataset."""
    print("\n===== Data Visualization =====")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot distribution of values in each column
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df.columns):
        plt.subplot(2, 4, i+1)
        df[col].hist(bins=30)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/column_distributions.png")
    print(f"Saved column distributions to {output_dir}/column_distributions.png")
    
    # Generate correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    plt.matshow(correlation_matrix, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Correlation Matrix')
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    print(f"Saved correlation matrix to {output_dir}/correlation_matrix.png")
    
    return

def main():
    """Main function to execute the analysis."""
    # Load the data
    df = load_data()
    
    # Analyze the data
    df = analyze_data(df)
    
    # Visualize the data
    visualize_data(df)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 