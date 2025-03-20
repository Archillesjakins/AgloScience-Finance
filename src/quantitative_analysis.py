#!/usr/bin/env python3
"""
Quantitative Finance Analysis for 200k.txt

This script performs advanced quantitative finance analysis on the dataset,
including pattern recognition, time series analysis, and predictive modeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time
import seaborn as sns

def load_and_preprocess_data(file_path='200k.txt'):
    """Load and preprocess the data from the file."""
    print(f"Loading data from {file_path}...")
    start_time = time.time()
    
    # Read the data - semicolon separated values with no header
    df = pd.read_csv(file_path, header=None, sep=';')
    
    # Assign column names based on their position
    df.columns = [f'col_{i}' for i in range(len(df.columns))]
    
    # Report on the loading time
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")
    print(f"Dataset shape: {df.shape}")
    
    # Clean the data - handle missing values and outliers
    # Assuming the last column might be empty due to trailing semicolon
    if df[df.columns[-1]].isnull().all():
        df = df.drop(df.columns[-1], axis=1)
        print(f"Dropped empty last column. New shape: {df.shape}")
    
    # Check and handle any remaining missing values
    if df.isnull().any().any():
        print("Filling missing values...")
        # Fill numeric missing values with median of the column
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    
    return df

def detect_patterns(df):
    """Detect patterns in the data that might be relevant for trading."""
    print("\n===== Pattern Detection =====")
    
    # Generate a correlation matrix to identify relationships
    correlation_matrix = df.corr()
    
    # Find highest correlations (excluding self-correlations)
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.5:  # Threshold for "high" correlation
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                        correlation_matrix.columns[j], 
                                        corr))
    
    print("\nHigh correlation pairs:")
    for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"{col1} and {col2}: {corr:.4f}")
    
    # Visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig("output/correlation_heatmap.png")
    print("Saved correlation heatmap to output/correlation_heatmap.png")
    
    return high_corr_pairs

def build_predictive_model(df):
    """Build a predictive model for one of the columns using others as features."""
    print("\n===== Predictive Modeling =====")
    
    # Select a target column (assuming we're trying to predict col_0 using others)
    target_column = 'col_0'
    
    # Features are all columns except the target
    feature_columns = [col for col in df.columns if col != target_column]
    
    # Print prediction task
    print(f"Building a model to predict {target_column} using {', '.join(feature_columns)}")
    
    # Prepare features and target
    X = df[feature_columns].values
    y = df[target_column].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train a Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Feature importance
    print("\nFeature importance:")
    feature_importance = list(zip(feature_columns, model.feature_importances_))
    for feature, importance in sorted(feature_importance, key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(model.feature_importances_)
    plt.barh(range(len(sorted_idx)), model.feature_importances_[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_columns[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Prediction')
    plt.tight_layout()
    plt.savefig("output/feature_importance.png")
    print("Saved feature importance plot to output/feature_importance.png")
    
    return model

def perform_time_series_analysis(df):
    """Perform time series analysis assuming the data is sequential."""
    print("\n===== Time Series Analysis =====")
    
    # Assuming the data might represent sequential observations
    # Let's analyze the autocorrelation for the first column
    
    from pandas.plotting import autocorrelation_plot
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    target_col = 'col_0'
    
    print(f"Performing autocorrelation analysis on {target_col}")
    
    plt.figure(figsize=(12, 6))
    autocorrelation_plot(df[target_col])
    plt.title(f'Autocorrelation Plot for {target_col}')
    plt.tight_layout()
    plt.savefig("output/autocorrelation.png")
    print("Saved autocorrelation plot to output/autocorrelation.png")
    
    # ACF and PACF plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(df[target_col], ax=axes[0], lags=40)
    axes[0].set_title(f'Autocorrelation Function for {target_col}')
    
    plot_pacf(df[target_col], ax=axes[1], lags=40)
    axes[1].set_title(f'Partial Autocorrelation Function for {target_col}')
    
    plt.tight_layout()
    plt.savefig("output/acf_pacf.png")
    print("Saved ACF and PACF plots to output/acf_pacf.png")
    
    return

def main():
    """Main function to execute the quantitative finance analysis."""
    # Create output directory
    Path("output").mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess the data
    df = load_and_preprocess_data()
    
    # Detect patterns in the data
    high_corr_pairs = detect_patterns(df)
    
    # Build and evaluate a predictive model
    model = build_predictive_model(df)
    
    # Perform time series analysis
    perform_time_series_analysis(df)
    
    print("\nQuantitative finance analysis completed successfully!")

if __name__ == "__main__":
    main() 