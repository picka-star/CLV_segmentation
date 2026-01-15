import pandas as pd
import numpy as np
import streamlit as st

class RFMAnalyzer:
    def __init__(self):
        pass
    
    def calculate_rfm(self, df):
        """Calculate RFM scores"""
        # Reference date for recency calculation
        reference_date = df['Transaction_Date'].max() + pd.Timedelta(days=1)
        
        # Calculate RFM
        rfm = df.groupby('CustomerID').agg(
            Recency=('Transaction_Date', lambda x: (reference_date - x.max()).days),
            Frequency=('Transaction_ID', 'nunique'),
            Monetary=('Total_Price', 'sum')
        ).reset_index()
        
        # Calculate scores (1-5)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])
        
        # Convert to numeric
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Combined scores
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        
        return rfm, None
    
    def analyze_product_categories(self, df, rfm):
        """Analyze product category proportions"""
        # Calculate category proportions
        category_purchase = df.groupby(['CustomerID', 'Product_Category'])['Quantity'].sum().reset_index()
        total_per_customer = category_purchase.groupby('CustomerID')['Quantity'].sum().reset_index()
        total_per_customer.columns = ['CustomerID', 'Total_Quantity']
        
        # Merge and calculate proportions
        category_prop = pd.merge(category_purchase, total_per_customer, on='CustomerID')
        category_prop['Proportion'] = category_prop['Quantity'] / category_prop['Total_Quantity']
        
        # Pivot table
        category_pivot = category_prop.pivot_table(
            index='CustomerID',
            columns='Product_Category',
            values='Proportion',
            fill_value=0
        ).reset_index()
        
        # Merge with RFM
        customer_data = pd.merge(rfm, category_pivot, on='CustomerID', how='left')
        
        return customer_data