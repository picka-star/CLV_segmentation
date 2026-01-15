import numpy as np
import pandas as pd

class AHPCalculator:
    """Analytical Hierarchy Process for weighting criteria"""
    
    def __init__(self):
        pass
    
    def calculate_weights(self, criteria_matrix):
        """Calculate weights using AHP"""
        # Normalize the matrix
        normalized = criteria_matrix / criteria_matrix.sum(axis=0)
        
        # Calculate weights
        weights = normalized.mean(axis=1)
        
        return weights