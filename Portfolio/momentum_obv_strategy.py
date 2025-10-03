#!/usr/bin/env python
# coding: utf-8

# In[1]:


# File: momentum_obv_strategy.py

# File: momentum_obv_strategy.py

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Tuple
from Base_Strategy import BaseStrategy

class MomentumOBVStrategy(BaseStrategy):
    """
    A strategy that combines a momentum indicator with On-Balance Volume (OBV).
    
    Strategy Logic:
    - Long Entry: Momentum is positive AND OBV is rising.
    - Short Entry: Momentum is negative.
    - Exits: An entry in the opposite direction serves as an exit signal.
    """
    
    def __init__(self, name: str, symbol: str, **kwargs):
        super().__init__(name, symbol, **kwargs)
        
        # Strategy-specific parameters
        self.mom_period = kwargs.get('mom_period', 75)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates On-Balance Volume (OBV) and Momentum."""
        df_indic = df.copy()

        # Calculate OBV using np.sign for robust vectorization
        price_change_sign = np.sign(df_indic['Close'].diff())
        signed_volume = price_change_sign * df_indic['Volume']
        df_indic['OBV'] = signed_volume.cumsum().fillna(0)
        
        # Calculate Momentum
        df_indic['Momentum'] = df_indic['Close'].diff(periods=self.mom_period)
        
        return df_indic.dropna()
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Generates entry and exit signals based on Momentum and OBV."""
        
        momentum_positive = df['Momentum'] > 0
        momentum_negative = df['Momentum'] < 0
        obv_rising = df['OBV'] > df['OBV'].shift(1)
        
        entries = momentum_positive & obv_rising
        short_entries = momentum_negative
        
        exits = short_entries
        short_exits = entries
        
        return entries, exits, short_entries, short_exits


# In[ ]:




