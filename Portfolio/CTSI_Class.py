#!/usr/bin/env python
# coding: utf-8

# In[4]:

import pandas as pd
import yfinance as yf
import vectorbt as vbt
from typing import Tuple
from Base_Strategy import BaseStrategy
import plotly.graph_objects as go


# In[5]:


class MomentumCryptoACStrategy(BaseStrategy):
    """
    Momentum Crypto Strategy using Accelerator Oscillator (AC) indicator.
    
    Strategy Logic:
    - Long when AO > AC
    - Short when AO < AC
    - Uses AC indicator which is derived from Awesome Oscillator (AO)
    """
    
    def __init__(self, name: str, symbol: str = 'BTC-USD', **kwargs):
        super().__init__(name, symbol, **kwargs)
        
        # AC Indicator parameters
        self.short_sma_period = kwargs.get('short_sma_period', 5)
        self.long_sma_period = kwargs.get('long_sma_period', 34)
        self.ao_sma_period = kwargs.get('ao_sma_period', 5)
    
    def calculate_ac_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Accelerator Oscillator (AC) from a DataFrame with 'High' and 'Low' columns.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        
        Returns:
            pd.DataFrame: DataFrame with AC indicator added.
        """
        df = df.copy()
        
        # Calculate Median Price
        df['Median_Price'] = (df['High'] + df['Low']) / 2
        
        # Calculate short and long period SMAs of the Median Price
        df['SMA_Short'] = df['Median_Price'].rolling(window=self.short_sma_period).mean()
        df['SMA_Long'] = df['Median_Price'].rolling(window=self.long_sma_period).mean()
        
        # Calculate Awesome Oscillator (AO)
        df['AO'] = df['SMA_Short'] - df['SMA_Long']
        
        # Calculate SMA of the AO
        df['SMA_AO'] = df['AO'].rolling(window=self.ao_sma_period).mean()
        
        # Calculate Accelerator Oscillator (AC)
        df['AC'] = df['AO'] - df['SMA_AO']
        
        # Drop NaN values (due to the rolling averages)
        df = df.dropna()
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators"""
        return self.calculate_ac_indicator(df)
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate entry/exit signals for long and short positions.
        
        Returns:
            Tuple of (entries, exits, short_entries, short_exits)
        """
        # Strategy conditions
        entries = df['AO'] > df['AC']  # Long when Awesome > Accelerator
        exits = df['AO'] < df['AC']    # Exit long when AO < AC
        short_entries = df['AO'] < df['AC']  # Short when AO < AC
        short_exits = df['AO'] > df['AC']    # Exit short when AO > AC
        
        return entries, exits, short_entries, short_exits
    
    def plot_ac_indicator(self, show_candlestick: bool = True):
        """
        Plot the Accelerator Oscillator (AC) values as bars using Plotly.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df_with_indicators = self.calculate_indicators(self.data.copy())
        ac_values = df_with_indicators['AC']
        
        # Initialize the list for bar colors
        colors = []
        # Loop through the AC values and determine the color (green for increase, red for decrease)
        for i in range(1, len(ac_values)):
            if ac_values.iloc[i] > ac_values.iloc[i-1]:  # AC is increasing
                colors.append("green")
            else:  # AC is decreasing
                colors.append("red")
        # Add a color for the first value (no previous value to compare to)
        colors.insert(0, "gray")  # Gray color for the first bar
        
        # Create a Plotly figure with subplots
        from plotly.subplots import make_subplots
        
        if show_candlestick:
            fig = make_subplots(rows=2, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.1,
                              subplot_titles=('Price Chart', 'Accelerator Oscillator'))
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df_with_indicators.index,
                open=df_with_indicators['Open'],
                high=df_with_indicators['High'],
                low=df_with_indicators['Low'],
                close=df_with_indicators['Close'],
                name=self.symbol
            ), row=1, col=1)
            
            # Add AC indicator
            fig.add_trace(go.Bar(
                x=ac_values.index,
                y=ac_values,
                marker=dict(color=colors),
                name="Accelerator Oscillator"
            ), row=2, col=1)
            
            # Add horizontal line at y=0 for reference
            fig.add_hline(y=0, line=dict(color="black", width=2, dash="dash"), row=2, col=1)
            
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ac_values.index,
                y=ac_values,
                marker=dict(color=colors),
                name="Accelerator Oscillator"
            ))
            fig.add_hline(y=0, line=dict(color="black", width=2, dash="dash"))
        
        # Update layout
        fig.update_layout(
            title=f"{self.name} - {self.symbol} AC Indicator",
            xaxis_title="Date",
            showlegend=True,
            template="plotly"
        )
        
        fig.show()
        
    def plot_trades(self):
        """
        Plot price chart with trades/positions overlaid.
        Equivalent to your original: 
        fig = BTC.vbt.plot(trace_kwargs=dict(name='Close'))
        pf.positions.plot(close_trace_kwargs=dict(visible=False), fig=fig)
        """
        if self.portfolio is None:
            raise ValueError("Portfolio not created. Run backtest() first.")
            # Get the close price series from the data used in backtesting
        df_with_indicators = self.calculate_indicators(self.data.copy())
        close_series = df_with_indicators['Close']
        
        # Create the base price plot
        fig = close_series.vbt.plot(trace_kwargs=dict(name='Close'))
        
        # Overlay positions on the price chart
        self.portfolio.positions.plot(close_trace_kwargs=dict(visible=False), fig=fig)
        
        # Update layout
        fig.update_layout(
            title=f"{self.name} - {self.symbol} Price with Trades/Positions"
        )
        
        fig.show()


# In[6]:


# Example usage and testing
if __name__ == "__main__":
    # Initialize strategy
    strategy = MomentumCryptoACStrategy(symbol='BTC-USD')
    
    # Load data
    strategy.load_data(start_date='2019-01-07')
    
    # Run backtest
    portfolio = strategy.backtest(
        init_cash=15000,
        fees=0,
        sl_stop=0.02
    )
    
    # Get performance metrics
    performance = strategy.get_performance_metrics()
    print("Performance Metrics:")
    for key, value in performance.items():
        if key != 'portfolio':
            print(f"{key}: {value}")
    
    # Plot results
    strategy.plot_results()
    #Plot trades
    strategy.plot_trades()
    # Plot AC indicator
    strategy.plot_ac_indicator()
    
    # Access the vectorbt portfolio for detailed analysis
    print("\nDetailed Statistics:")
    print(portfolio.stats())


# In[ ]:




