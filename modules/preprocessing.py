import pandas as pd
import numpy as np
import streamlit as st

class DataPreprocessor:
    def __init__(self):
        pass
    
    def preprocess(self, df):
        """Main preprocessing function"""
        df_clean = df.copy()
        
        # Convert date columns
        date_cols = [col for col in df_clean.columns if 'date' in col.lower()]
        for col in date_cols:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Handle missing values
        critical_cols = ['CustomerID', 'Transaction_ID', 'Transaction_Date', 
                        'Product_Category', 'Quantity', 'Avg_Price']
        critical_cols = [col for col in critical_cols if col in df_clean.columns]
        
        df_clean = df_clean.dropna(subset=critical_cols)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Filter invalid values
        if 'Quantity' in df_clean.columns:
            df_clean = df_clean[df_clean['Quantity'] > 0]
        if 'Avg_Price' in df_clean.columns:
            df_clean = df_clean[df_clean['Avg_Price'] > 0]
        
        # Create derived variables
        df_clean['Total_Price'] = df_clean['Quantity'] * df_clean['Avg_Price']
        
        # Normalize product categories
        if 'Product_Category' in df_clean.columns:
            df_clean['Product_Category'] = df_clean['Product_Category'].str.strip().str.lower()
        
        return df_clean