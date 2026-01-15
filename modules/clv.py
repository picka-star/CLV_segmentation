import pandas as pd
import numpy as np

class CLVCalculator:
    """Customer Lifetime Value Calculator"""
    
    def __init__(self):
        pass
    
    def calculate_clv(self, rfm_data):
        """Calculate CLV based on RFM"""
        # Simple CLV calculation: Monetary * (Frequency / Recency) * retention_rate
        rfm_data['CLV'] = rfm_data['Monetary'] * (rfm_data['Frequency'] / (rfm_data['Recency'] + 1)) * 0.2
        return rfm_data