#!/usr/bin/env python3
"""
Trading Strategy Implementation for 200k.txt

This script implements a simple trading strategy based on patterns identified in the dataset.
It simulates trades and evaluates the performance of the strategy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

class TradingStrategy:
    """Class implementing a trading strategy based on the dataset."""
    
    def __init__(self, data_file='200k.txt', model_path=None):
        """Initialize the trading strategy with data and optional pre-trained model."""
        self.data_file = data_file
        self.model_path = model_path
        self.model = None
        self.df = None
        self.scaler = None
        self.performance_metrics = {}
        
    def load_data(self):
        """Load the dataset."""
        print(f"Loading data from {self.data_file}...")
        start_time = time.time()
        
        # Read the data - semicolon separated values with no header
        self.df = pd.read_csv(self.data_file, header=None, sep=';')
        
        # Assign column names based on their position
        self.df.columns = [f'col_{i}' for i in range(len(self.df.columns))]
        
        # Handle empty last column due to trailing semicolon if present
        if self.df[self.df.columns[-1]].isnull().all():
            self.df = self.df.drop(self.df.columns[-1], axis=1)
        
        # Report on the loading time
        load_time = time.time() - start_time
        print(f"Data loaded in {load_time:.2f} seconds")
        print(f"Dataset shape: {self.df.shape}")
        
        return self.df
        
    def prepare_features(self, target_col='col_0', test_size=0.25):
        """Prepare features for the predictive model."""
        if self.df is None:
            self.load_data()
            
        # Define features and target
        self.target_column = target_col
        self.feature_columns = [col for col in self.df.columns if col != self.target_column]
        
        # Prepare features and target
        X = self.df[self.feature_columns].values
        y = self.df[self.target_column].values
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, n_estimators=100):
        """Train a predictive model or load a pre-trained model."""
        if self.model_path and Path(self.model_path).exists():
            print(f"Loading pre-trained model from {self.model_path}")
            self.model = joblib.load(self.model_path)
        else:
            print("Training new model...")
            self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Save the model for future use
            if self.model_path:
                print(f"Saving model to {self.model_path}")
                Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.model, self.model_path)
                
        return self.model
    
    def generate_signals(self, X_test):
        """Generate trading signals based on the predictive model."""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before generating signals")
        
        # Predict using the model
        predictions = self.model.predict(X_test)
        
        # Convert predictions to trading signals
        # This is a simple approach; more sophisticated signal generation can be implemented
        signals = pd.Series(predictions, name='signal')
        
        # Map the signals to actions:
        # Buy (1) when prediction is greater than the median
        # Sell (0) when prediction is less than or equal to the median
        median_prediction = np.median(predictions)
        signals = signals.apply(lambda x: 1 if x > median_prediction else 0)
        
        return signals
    
    def backtest_strategy(self, signals, initial_capital=10000.0):
        """Backtest the trading strategy with generated signals."""
        print("\n===== Backtesting Trading Strategy =====")
        
        # Create a positions DataFrame
        positions = pd.DataFrame(index=signals.index)
        positions['signal'] = signals
        
        # Calculate daily returns (for simplicity, using random values)
        # In a real scenario, you'd use actual price data
        # This simulates daily returns with mean 0 and std 0.01
        np.random.seed(42)  # For reproducibility
        daily_returns = pd.Series(np.random.normal(0, 0.01, len(signals)), index=signals.index)
        
        # Calculate strategy returns
        positions['returns'] = daily_returns
        positions['strategy_returns'] = positions['signal'].shift(1) * positions['returns']
        positions['strategy_returns'].fillna(0, inplace=True)
        
        # Calculate cumulative returns
        positions['buy_hold_cumulative'] = (1 + positions['returns']).cumprod()
        positions['strategy_cumulative'] = (1 + positions['strategy_returns']).cumprod()
        
        # Calculate portfolio value
        positions['buy_hold_value'] = initial_capital * positions['buy_hold_cumulative']
        positions['strategy_value'] = initial_capital * positions['strategy_cumulative']
        
        # Calculate metrics
        self.performance_metrics = {
            'total_trades': positions['signal'].diff().abs().sum(),
            'profitable_trades': (positions['strategy_returns'] > 0).sum(),
            'final_portfolio_value': positions['strategy_value'].iloc[-1],
            'buy_hold_portfolio_value': positions['buy_hold_value'].iloc[-1],
            'strategy_return': positions['strategy_value'].iloc[-1] / initial_capital - 1,
            'buy_hold_return': positions['buy_hold_value'].iloc[-1] / initial_capital - 1,
        }
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        strategy_std = positions['strategy_returns'].std()
        if strategy_std > 0:
            self.performance_metrics['strategy_sharpe'] = (
                positions['strategy_returns'].mean() / strategy_std * np.sqrt(252)  # Annualized
            )
        else:
            self.performance_metrics['strategy_sharpe'] = 0
            
        buy_hold_std = positions['returns'].std()
        self.performance_metrics['buy_hold_sharpe'] = (
            positions['returns'].mean() / buy_hold_std * np.sqrt(252)  # Annualized
        )
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        for metric, value in self.performance_metrics.items():
            print(f"{metric}: {value}")
        
        # Visualize the strategy performance
        self._plot_strategy_performance(positions)
        
        return positions, self.performance_metrics
    
    def _plot_strategy_performance(self, positions):
        """Plot the performance of the strategy vs buy and hold."""
        plt.figure(figsize=(12, 8))
        
        # Plot the portfolio value
        plt.subplot(2, 1, 1)
        positions['buy_hold_value'].plot(color='blue', linestyle='-', label='Buy & Hold')
        positions['strategy_value'].plot(color='green', linestyle='-', label='Strategy')
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        
        # Plot the returns
        plt.subplot(2, 1, 2)
        positions['strategy_returns'].plot(color='red', label='Strategy Returns')
        plt.title('Daily Returns')
        plt.ylabel('Return (%)')
        plt.grid(True)
        
        # Plot the signals
        buy_signals = positions[positions['signal'] == 1].index
        sell_signals = positions[positions['signal'] == 0].index
        
        for buy_idx in buy_signals[:20]:  # Limit to first 20 for clarity
            plt.axvline(x=buy_idx, color='green', linestyle='--', alpha=0.3)
        for sell_idx in sell_signals[:20]:  # Limit to first 20 for clarity
            plt.axvline(x=sell_idx, color='red', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("output/strategy_performance.png")
        print("Saved strategy performance plot to output/strategy_performance.png")
        
        # Distribution of returns
        plt.figure(figsize=(10, 6))
        positions['strategy_returns'].hist(bins=50, alpha=0.5, label='Strategy Returns')
        positions['returns'].hist(bins=50, alpha=0.5, label='Market Returns')
        plt.title('Distribution of Returns')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(False)
        plt.savefig("output/returns_distribution.png")
        print("Saved returns distribution plot to output/returns_distribution.png")
        
        return

def main():
    """Main function to execute the trading strategy."""
    # Create output directory
    Path("output").mkdir(parents=True, exist_ok=True)
    
    # Initialize trading strategy
    strategy = TradingStrategy(data_file='200k.txt', model_path='output/trading_model.pkl')
    
    # Load data
    strategy.load_data()
    
    # Prepare features
    X_train, X_test, y_train, y_test = strategy.prepare_features()
    
    # Train or load model
    strategy.train_model(X_train, y_train)
    
    # Generate trading signals
    signals = strategy.generate_signals(X_test)
    
    # Backtest the strategy
    positions, metrics = strategy.backtest_strategy(signals)
    
    print("\nTrading strategy simulation completed successfully!")

if __name__ == "__main__":
    main() 