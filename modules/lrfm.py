import pandas as pd
import numpy as np

class LRFAnalyzer:
    """LRFM Analysis (Length, Recency, Frequency, Monetary)"""
    
    def __init__(self):
        pass
    
    def calculate_lrfm(self, df):
        """Calculate LRFM values"""
        # Length: Customer tenure
        # Recency: Days since last purchase
        # Frequency: Number of purchases
        # Monetary: Total spending
        
        lrfm = df.groupby('CustomerID').agg(
            Length=('Transaction_Date', lambda x: (x.max() - x.min()).days),
            Recency=('Transaction_Date', lambda x: (pd.Timestamp.now() - x.max()).days),
            Frequency=('Transaction_ID', 'nunique'),
            Monetary=('Total_Price', 'sum')
        ).reset_index()
        
        return lrfm