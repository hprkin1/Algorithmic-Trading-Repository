#!/usr/bin/env python
# coding: utf-8

# In[2]:


# File: portfolio_manager.py

import pandas as pd
import vectorbt as vbt
import time
from typing import List
from Base_Strategy import BaseStrategy
import numpy as np  # <-- IMPORT NUMPY

class PortfolioManager:
    """
    Manages a collection of trading strategies, generates their signals,
    and runs a single, unified backtest on the combined signals.
    """
    def __init__(self, strategies: List[BaseStrategy]):
        if not strategies:
            raise ValueError("Strategies list cannot be empty.")
        self.strategies = strategies
        self.combined_portfolio = None
        self.total_init_cash = 0

    def run(self, init_cash_per_strategy: float = 100000, **kwargs):
        """
        Generates signals for each strategy using its own parameters
        and runs one combined backtest.
        """
        print("--- Generating Signals for All Strategies ---")
        
        all_prices, all_entries, all_exits = [], [], []
        all_short_entries, all_short_exits = [], []
        all_sl_stops = []

        for i, strategy in enumerate(self.strategies):
            print(f"Processing Strategy {i+1}/{len(self.strategies)}: {strategy.name}")
            try:
                start_date = strategy.config.get('start_date', '2019-01-01')
                print(f"  > Using start date: {start_date}")
                
                if i > 0:
                    print("  > Pausing for 2 seconds before next download...")
                    time.sleep(2)
                
                df_data = strategy.load_data(start_date=start_date)
                df_indic = strategy.calculate_indicators(df_data)
                entries, exits, short_entries, short_exits = strategy.generate_signals(df_indic)

                symbols = strategy.symbols if hasattr(strategy, 'symbols') else [strategy.symbol]
                
                if len(symbols) == 1:
                    prices = df_indic[['Close']].rename(columns={'Close': symbols[0]})
                else:
                    prices = df_indic[symbols]

                sl_value = strategy.config.get('sl_stop')
                sl_to_add = sl_value if sl_value is not None else np.nan
                
                for s in symbols:
                    all_sl_stops.append(sl_to_add)

                all_prices.append(prices)
                all_entries.append(entries)
                all_exits.append(exits)
                all_short_entries.append(short_entries)
                all_short_exits.append(short_exits)
            except Exception as e:
                print(f"  > FAILED to process strategy '{strategy.name}'. Error: {e}")

        if not all_prices:
            raise ValueError("No valid signals were generated from any strategy.")

        print("\n--- Combining All Prices and Signals ---")
        final_prices = pd.concat(all_prices, axis=1)
        
        # FIX for FutureWarning: Convert to float before filling, then to bool
        final_entries = pd.concat(all_entries, axis=1).astype(float).fillna(False).astype(bool)
        final_exits = pd.concat(all_exits, axis=1).astype(float).fillna(False).astype(bool)
        final_short_entries = pd.concat(all_short_entries, axis=1).astype(float).fillna(False).astype(bool)
        final_short_exits = pd.concat(all_short_exits, axis=1).astype(float).fillna(False).astype(bool)

        self.total_init_cash = init_cash_per_strategy * len(all_prices)
        print(f"Total Initial Capital: ${self.total_init_cash:.2f}")

        print("Running combined portfolio backtest...")
        
        self.combined_portfolio = vbt.Portfolio.from_signals(
            final_prices,
            entries=final_entries,
            exits=final_exits,
            short_entries=final_short_entries,
            short_exits=final_short_exits,
            freq='1D',
            init_cash=self.total_init_cash,
            sl_stop=all_sl_stops,
            **kwargs
        )

    def get_stats(self):
        if self.combined_portfolio is None: raise ValueError("Portfolio not created.")
        return self.combined_portfolio.stats()

    def plot(self):
        if self.combined_portfolio is None: raise ValueError("Portfolio not created.")
        print("Plotting combined portfolio equity curve in percentage terms...")

        # --- Portfolio cumulative return ---
        value_df = self.combined_portfolio.value()
        total_value = value_df.sum(axis=1)
        cumulative_return_pct = (total_value / self.total_init_cash) - 1
        
        # --- Combined Benchmark cumulative return (Buy-and-Hold Equally-Weighted) ---
        benchmark_returns_df = self.combined_portfolio.benchmark_returns()
        cumulative_benchmark_growth = (1 + benchmark_returns_df).cumprod()
        avg_cumulative_growth = cumulative_benchmark_growth.mean(axis=1)
        cumulative_benchmark_ret = avg_cumulative_growth - 1
        
        # --- Plot both series on the same figure ---
        fig = cumulative_return_pct.vbt.plot(
            trace_kwargs=dict(name='Portfolio'),
            title="Combined Portfolio Performance", 
            yaxis_title="Cumulative Return",
            yaxis_tickformat='.2%'
        )

        cumulative_benchmark_ret.vbt.plot(
            trace_kwargs=dict(name='Benchmark (Equal-Weight)'),
            fig=fig
        )
        
        fig.show()







# In[ ]:




