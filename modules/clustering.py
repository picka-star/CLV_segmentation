import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st

class CustomerClustering:
    def __init__(self):
        pass
    
    def perform_clustering(self, customer_data, n_clusters=4, use_category=True):
        """Perform K-Means clustering"""
        # Prepare features
        features = ['R_Score', 'F_Score', 'M_Score']
        
        # Add product categories if requested
        if use_category:
            category_cols = [col for col in customer_data.columns 
                           if col not in ['CustomerID', 'Recency', 'Frequency', 'Monetary',
                                         'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'RFM_Total', 'Segment']]
            
            if len(category_cols) > 0:
                # Take top categories by variance
                category_variance = customer_data[category_cols].var().sort_values(ascending=False)
                top_categories = category_variance.head(5).index.tolist()
                features.extend(top_categories)
        
        # Prepare data
        X = customer_data[features].fillna(0)
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        customer_data['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels
        customer_data['Cluster_Label'] = customer_data['Cluster'].apply(
            lambda x: f'Klaster {x}'
        )
        
        # Create cluster profile
        cluster_profile = self.create_cluster_profile(customer_data)
        
        return customer_data, cluster_profile
    
    def create_cluster_profile(self, customer_data):
        """Create detailed cluster profile"""
        cluster_profile = customer_data.groupby('Cluster').agg({
            'Recency': ['mean', 'std', 'min', 'max'],
            'Frequency': ['mean', 'std', 'min', 'max'],
            'Monetary': ['mean', 'std', 'min', 'max'],
            'R_Score': 'mean',
            'F_Score': 'mean',
            'M_Score': 'mean',
            'RFM_Total': 'mean',
            'CustomerID': 'count'
        }).round(2)
        
        # Flatten column names
        cluster_profile.columns = ['_'.join(col).strip() for col in cluster_profile.columns.values]
        
        # Rename columns
        rename_dict = {
            'Recency_mean': 'Avg_Recency',
            'Recency_std': 'Std_Recency',
            'Frequency_mean': 'Avg_Frequency',
            'Frequency_std': 'Std_Frequency',
            'Monetary_mean': 'Avg_Monetary',
            'Monetary_std': 'Std_Monetary',
            'R_Score_mean': 'Avg_R_Score',
            'F_Score_mean': 'Avg_F_Score',
            'M_Score_mean': 'Avg_M_Score',
            'RFM_Total_mean': 'Avg_RFM_Total',
            'CustomerID_count': 'Jumlah_Pelanggan'
        }
        
        cluster_profile = cluster_profile.rename(columns=rename_dict)
        
        # Add percentage
        total_customers = cluster_profile['Jumlah_Pelanggan'].sum()
        cluster_profile['Persentase'] = (cluster_profile['Jumlah_Pelanggan'] / total_customers * 100).round(2)
        
        # Reorder columns
        cluster_profile = cluster_profile[[
            'Jumlah_Pelanggan', 'Persentase',
            'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score', 'Avg_RFM_Total',
            'Avg_Recency', 'Std_Recency',
            'Avg_Frequency', 'Std_Frequency',
            'Avg_Monetary', 'Std_Monetary'
        ]]
        
        return cluster_profile