# Algorithmic Sciences - Quantitative Finance Project

This project contains a set of Python scripts for analyzing and implementing trading strategies based on the provided dataset (200k.txt).

## Project Structure

```
.
├── README.md                 # This documentation
├── requirements.txt          # Python package dependencies
├── 200k.txt                  # Main dataset file (271,100 records)
├── src/                      # Source code directory
│   ├── main.py               # Main entry point script
│   ├── analyze_data.py       # Basic data analysis script
│   ├── quantitative_analysis.py # Advanced quantitative analysis 
│   └── trading_strategy.py   # Trading strategy implementation
└── output/                   # Generated output (plots, models, etc.)
```

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running All Analysis Components

The simplest way to run all analysis components is to use the main script:

```bash
python3 src/main.py
```

This will run all three components in sequence:
1. Basic data analysis
2. Quantitative finance analysis
3. Trading strategy simulation

You can also run specific components:

```bash
# Run only basic analysis
python3 src/main.py --basic

# Run only quantitative analysis
python3 src/main.py --quant

# Run only trading strategy
python3 src/main.py --trading
```

### Basic Data Analysis

To run basic data analysis and generate statistical insights:

```bash
python3 src/analyze_data.py
```

This script:
- Loads the data from 200k.txt
- Calculates basic statistics for each column
- Checks for missing values and unique values
- Visualizes the distribution of values in each column
- Generates a correlation matrix

### Quantitative Finance Analysis

To perform advanced quantitative analysis:

```bash
python3 src/quantitative_analysis.py
```

This script:
- Loads and preprocesses the data
- Detects patterns and correlations relevant for trading
- Builds a predictive model using Random Forest
- Performs time series analysis on the data

### Trading Strategy Simulation

To implement and backtest a trading strategy:

```bash
python3 src/trading_strategy.py
```

This script:
- Implements a class for trading strategy based on the dataset
- Trains a predictive model to generate trading signals
- Backtests the strategy against simulated market data
- Calculates performance metrics like Sharpe ratio
- Visualizes the strategy performance

## Results

The analysis results and visualizations are saved in the `output/` directory:
- Column distributions
- Correlation matrix
- Feature importance for prediction
- Autocorrelation plots
- Strategy performance charts
- Returns distribution

## Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn
- statsmodels

## License

This project is part of the Algorithmic Sciences introductory task. 