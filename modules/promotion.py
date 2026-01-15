import pandas as pd
import numpy as np
import streamlit as st

class PromotionStrategy:
    def __init__(self):
        pass
    
    def generate_strategies(self, customer_data, cluster_profile, apriori_results):
        """Generate promotion strategies"""
        st.write("### 🎯 Strategi Promosi Berdasarkan Segmentasi")
        
        for cluster_num, profile in cluster_profile.iterrows():
            st.write(f"#### 📊 Klaster {cluster_num}")
            
            # Determine cluster type
            if profile['Avg_R_Score'] >= 4 and profile['Avg_F_Score'] >= 4 and profile['Avg_M_Score'] >= 4:
                cluster_type = "🟢 Premium"
                strategies = [
                    "Program loyalty eksklusif",
                    "Personalized recommendations",
                    "VIP customer service"
                ]
            elif profile['Avg_R_Score'] >= 4 and profile['Avg_F_Score'] >= 3:
                cluster_type = "🔵 Loyal"
                strategies = [
                    "Cross-selling opportunities",
                    "Bundle discounts",
                    "Upgrade promotions"
                ]
            elif profile['Avg_R_Score'] <= 2 and profile['Avg_F_Score'] >= 3:
                cluster_type = "🟡 Berisiko"
                strategies = [
                    "Re-activation campaigns",
                    "Win-back offers",
                    "Special discounts"
                ]
            else:
                cluster_type = "⚪ Reguler"
                strategies = [
                    "Regular promotions",
                    "Seasonal campaigns",
                    "Minimum purchase bonuses"
                ]
            
            # Display cluster info
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Jumlah Pelanggan", int(profile['Jumlah_Pelanggan']))
                st.metric("Persentase", f"{profile['Persentase']}%")
            
            with col2:
                st.write(f"**Tipe:** {cluster_type}")
                st.write("**Strategi:**")
                for strategy in strategies:
                    st.write(f"- {strategy}")
            
            # Add bundling recommendations if available
            if cluster_num in apriori_results and len(apriori_results[cluster_num]) > 0:
                st.write("**Rekomendasi Bundling:**")
                top_rules = apriori_results[cluster_num].head(2)
                for idx, row in top_rules.iterrows():
                    antecedents = ', '.join(list(row['antecedents']))
                    consequents = ', '.join(list(row['consequents']))
                    st.write(f"- {antecedents} → {consequents}")
            
            st.write("---")