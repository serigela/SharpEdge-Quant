#!/usr/bin/env python3
"""
Signal generation module for quant trading platform.
Uses ML models to predict price movements.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('signal_generation')

class FeatureEngineering:
    """Generate features from price data for ML models."""
    
    @staticmethod
    def add_technical_indicators(df):
        """Add technical indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Ensure we have the expected columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Missing required columns. Expected: {required_cols}, Got: {data.columns}")
            return data
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        
        # Moving averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        
        # Relative Strength Index (simplified)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['MA20_std'] = data['Close'].rolling(window=20).std()
        data['Upper_Band'] = data['MA20'] + (data['MA20_std'] * 2)
        data['Lower_Band'] = data['MA20'] - (data['MA20_std'] * 2)
        
        # MACD
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Remove NaN values
        data = data.dropna()
        
        return data
    
    @staticmethod
    def create_target(df, lookahead=1, threshold=0.0):
        """Create target variable for prediction.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            lookahead (int): Number of days to look ahead for prediction
            threshold (float): Threshold for determining positive class
            
        Returns:
            pd.DataFrame: DataFrame with target variable
        """
        data = df.copy()
        
        # Calculate future returns
        data['Future_Returns'] = data['Close'].pct_change(periods=lookahead).shift(-lookahead)
        
        # The final lookahead rows have no known outcome and must not be labeled 0.
        data = data.dropna(subset=['Future_Returns'])

        # Create binary target
        data['Target'] = np.where(data['Future_Returns'] > threshold, 1, 0)
        
        return data

class SignalGenerator:
    """Generate trading signals using ML models."""
    
    def __init__(self, model_type='random_forest', output_dir='data'):
        """Initialize the signal generator.
        
        Args:
            model_type (str): Type of ML model to use ('random_forest' or 'logistic_regression')
            output_dir (str): Directory to store output CSV files
        """
        self.model_type = model_type
        self.output_dir = output_dir
        self.model = None
        self.scaler = StandardScaler()
        os.makedirs(output_dir, exist_ok=True)
        
    def _create_model(self):
        """Create the ML model based on model_type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                C=1.0, 
                max_iter=1000, 
                random_state=42
            )
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return None
            
    def prepare_data(self, price_data):
        """Prepare data for model training.
        
        Args:
            price_data (pd.DataFrame): DataFrame with price data
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, feature_names
        """
        # Add technical indicators
        data = FeatureEngineering.add_technical_indicators(price_data)
        
        # Create target variable
        data = FeatureEngineering.create_target(data)
        
        # Define features
        feature_cols = [
            'Returns', 'MA5', 'MA10', 'MA20', 'RSI', 
            'Upper_Band', 'Lower_Band', 'MACD', 'Signal_Line'
        ]
        
        # Check if all feature columns exist
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing feature columns: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in data.columns]
        
        # Split features and target
        X = data[feature_cols]
        y = data['Target']
        
        if len(data) < 30 or not data.index.is_monotonic_increasing:
            raise ValueError('Need at least 30 chronologically ordered feature rows')
        split = int(len(data) * 0.8)
        # Purge the last training label: its future close falls in the test period.
        train_end = split - 1
        self.test_start = data.index[split]
        self.feature_cols = feature_cols
        X_train = self.scaler.fit_transform(X.iloc[:train_end])
        X_test = self.scaler.transform(X.iloc[split:])
        y_train, y_test = y.iloc[:train_end], y.iloc[split:]
        
        return X_train, X_test, y_train, y_test, feature_cols
        
    def train(self, price_data):
        """Train the ML model.
        
        Args:
            price_data (pd.DataFrame): DataFrame with price data
            
        Returns:
            object: Trained model
        """
        # Create model
        self.model = self._create_model()
        if self.model is None:
            return None
            
        # Prepare data
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(price_data)
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.holdout_metrics = {'accuracy': float(accuracy), 'samples': int(len(y_test)),
                                'start': str(self.test_start)}
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        return self.model
        
    def generate_signals(self, price_data):
        """Generate trading signals using the trained model.
        
        Args:
            price_data (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return None
            
        # Add technical indicators
        data = FeatureEngineering.add_technical_indicators(price_data)
        
        if not hasattr(self, 'test_start'):
            raise ValueError('Call train() before generating held-out signals')
        # Only held-out feature rows are scored; execution uses the next bar.
        eligible = data.index >= self.test_start
        positions = np.flatnonzero(eligible)[:-1]
        if not len(positions):
            raise ValueError('No next-bar held-out prices available')
        X = data.iloc[positions][self.feature_cols]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        signals = self.model.predict(X_scaled)
        
        execution = data.iloc[positions + 1]
        output = pd.DataFrame({
            'timestamp': execution.index,
            'price': execution['Open'].to_numpy(),
            'signal': signals
        })
        
        return output
    
    def save_signals(self, signals, ticker, filename=None):
        """Save the generated signals to a CSV file.
        
        Args:
            signals (pd.DataFrame): DataFrame with signals
            ticker (str): Stock ticker symbol
            filename (str, optional): Custom filename
            
        Returns:
            str: Path to the saved CSV file
        """
        if signals is None or signals.empty:
            logger.error("No signals to save")
            return None
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{ticker}_signals_{timestamp}.csv"
            
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate the full path
        output_path = os.path.join(self.output_dir, filename)
        
        # Save the data
        try:
            signals.to_csv(output_path, index=False)
            logger.info(f"Signals saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving signals: {str(e)}")
            return None

def main():
    """Main function to run signal generation from command line."""
    parser = argparse.ArgumentParser(description='Generate trading signals')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with price data')
    parser.add_argument('--model', type=str, default='random_forest', choices=['random_forest', 'logistic_regression'], help='ML model to use')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--output', type=str, help='Output file name')
    args = parser.parse_args()
    
    # Load price data
    try:
        price_data = pd.read_csv(args.input, index_col=0, parse_dates=True)
    except Exception as e:
        logger.error(f"Error loading price data: {str(e)}")
        return
    
    # Create signal generator
    generator = SignalGenerator(args.model)
    
    # Train model
    generator.train(price_data)
    
    # Generate signals
    signals = generator.generate_signals(price_data)
    
    # Save signals
    if signals is not None:
        generator.save_signals(signals, args.ticker, args.output)

if __name__ == "__main__":
    main()
