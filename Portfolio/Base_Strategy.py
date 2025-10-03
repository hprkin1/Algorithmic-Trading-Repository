#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import yfinance as yf
import vectorbt as vbt
import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Provides a unified interface for backtesting and portfolio management.
    """
    
    def __init__(self, name: str, symbol: str, **kwargs):
        self.name = name
        self.symbol = symbol
        self.data = None
        self.portfolio = None
        self.config = kwargs
        
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators"""
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Generate entry/exit signals for long and short positions"""
        pass
    
    def load_data(self, start_date: str = '2019-01-01', interval: str = '1d') -> pd.DataFrame:
        """Load price data from Yahoo Finance with a retry mechanism."""
        max_retries = 3
        retry_delay = 5 # seconds
        
        for attempt in range(max_retries):
            try:
                symbols_to_download = self.symbols if hasattr(self, 'symbols') else self.symbol
                print(f"  > Attempting download for {symbols_to_download} (Attempt {attempt + 1}/{max_retries})...")
                
                df = yf.download(symbols_to_download, interval=interval, start=start_date)
                
                if df.empty:
                    raise ValueError(f"yfinance returned an empty DataFrame for {self.symbol}.")
                
                print(f"  > Download for {self.symbol} successful.")
                self.data = df
                return df

            except Exception as e:
                print(f"  > Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"  > Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"  > Max retries reached. Could not download data for {self.symbol}.")
                    raise e

    def backtest(self, init_cash: float = 15000, fees: float = 0.0005, 
                 sl_stop: float = None, **kwargs) -> vbt.Portfolio:
        """Run backtest with unified parameters"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df_with_indicators = self.calculate_indicators(self.data.copy())
        entries, exits, short_entries, short_exits = self.generate_signals(df_with_indicators)
        
        # For multi-asset strategies, price_data should be the DataFrame of prices
        # For single-asset, it's just the 'Close' Series
        price_data = df_with_indicators[self.symbols] if hasattr(self, 'symbols') else df_with_indicators['Close']
        
        self.portfolio = vbt.Portfolio.from_signals(
            price_data,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            sl_stop=sl_stop,
            fees=fees,
            init_cash=init_cash,
            freq='1D',
            **kwargs
        )
        
        return self.portfolio

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get standardized performance metrics"""
        if self.portfolio is None:
            raise ValueError("Portfolio not created. Run backtest() first.")
        
        stats = self.portfolio.stats()
        return {
            'strategy_name': self.name,
            'symbol': self.symbol,
            'total_return': stats['Total Return [%]'],
            'annual_return': stats.get('Annual Return [%]', None),
            'max_drawdown': stats['Max Drawdown [%]'],
            'sharpe_ratio': stats.get('Sharpe Ratio', None),
            'win_rate': stats.get('Win Rate [%]', None),
            'total_trades': stats.get('# Trades', None),
            'final_value': stats['End Value'],
            'portfolio': self.portfolio
        }

    def plot_results(self):
        """
        Plots backtest results, handling both single and multi-asset portfolios.
        """
        if self.portfolio is None:
            raise ValueError("Portfolio not created. Run backtest() first.")
        
        # Check if the portfolio has a single column (single asset)
        if self.portfolio.wrapper.ndim == 1:
            fig = self.portfolio.plot()
            fig.update_layout(title=f"{self.name} - {self.symbol} Backtest Results")
            fig.show()
        else:
            # If multiple assets, loop through and plot each one
            for symbol in self.portfolio.wrapper.columns:
                print(f"\nGenerating portfolio plot for {symbol}...")
                fig = self.portfolio.plot(column=symbol)
                fig.update_layout(title=f"{self.name} - {symbol} Backtest Results")
                fig.show()

