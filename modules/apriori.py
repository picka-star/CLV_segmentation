import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st

class AprioriAnalyzer:
    def __init__(self):
        pass
    
    def analyze(self, df, customer_data, min_support=0.05, min_confidence=0.5, min_lift=1.2):
        """Perform Apriori analysis per cluster"""
        # Merge data with cluster information
        df_with_cluster = pd.merge(df, customer_data[['CustomerID', 'Cluster']], on='CustomerID')
        
        results = {}
        
        # Analyze for each cluster
        for cluster_num in sorted(customer_data['Cluster'].unique()):
            # Filter transactions for this cluster
            cluster_customers = customer_data[customer_data['Cluster'] == cluster_num]['CustomerID']
            cluster_transactions = df_with_cluster[df_with_cluster['CustomerID'].isin(cluster_customers)]
            
            if len(cluster_transactions) < 10:
                continue
            
            try:
                # Prepare basket data
                basket = self.prepare_basket_data(cluster_transactions)
                
                if basket is not None and len(basket) > 0:
                    # Apply Apriori algorithm
                    rules = self.apply_apriori(basket, min_support, min_confidence, min_lift)
                    
                    if len(rules) > 0:
                        results[cluster_num] = rules
                        
            except Exception as e:
                st.warning(f"⚠️ Error dalam analisis Apriori untuk klaster {cluster_num}: {str(e)}")
        
        return results
    
    def prepare_basket_data(self, transactions):
        """Prepare basket data for Apriori analysis"""
        try:
            basket = transactions.groupby(['Transaction_ID', 'Product_Category'])['Quantity'].sum().unstack().fillna(0)
            basket_binary = basket.applymap(lambda x: 1 if x > 0 else 0)
            return basket_binary
        except:
            return None
    
    def apply_apriori(self, basket_binary, min_support, min_confidence, min_lift):
        """Apply Apriori algorithm"""
        try:
            frequent_itemsets = apriori(basket_binary, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) > 0:
                rules = association_rules(frequent_itemsets, metric="confidence", 
                                         min_threshold=min_confidence)
                rules = rules[rules['lift'] >= min_lift]
                
                # Sort by confidence and lift
                rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
                
                return rules
            else:
                return pd.DataFrame()
                
        except Exception as e:
            return pd.DataFrame()