import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class DataVisualizer:
    def __init__(self):
        sns.set_style("whitegrid")
        self.color_palette = px.colors.qualitative.Set3
        
    def plot_preprocessing_comparison(self, df_before, df_after):
        """Visualisasi perbandingan sebelum dan sesudah preprocessing"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Data quality sebelum
            fig1 = go.Figure()
            
            # Count missing values
            if df_before.isnull().sum().sum() > 0:
                missing_before = df_before.isnull().sum().sum()
                valid_before = df_before.shape[0] * df_before.shape[1] - missing_before
                
                fig1.add_trace(go.Pie(
                    labels=['Data Valid', 'Missing Values'],
                    values=[valid_before, missing_before],
                    hole=0.3,
                    marker_colors=['#2ecc71', '#e74c3c']
                ))
                fig1.update_layout(title_text="Kualitas Data Sebelum Preprocessing")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("✅ Tidak ada missing values dalam data awal")
        
        with col2:
            # Data quality setelah
            fig2 = go.Figure()
            
            missing_after = df_after.isnull().sum().sum()
            valid_after = df_after.shape[0] * df_after.shape[1] - missing_after
            
            fig2.add_trace(go.Pie(
                labels=['Data Valid', 'Missing Values'],
                values=[valid_after, missing_after],
                hole=0.3,
                marker_colors=['#2ecc71', '#e74c3c']
            ))
            fig2.update_layout(title_text="Kualitas Data Setelah Preprocessing")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Statistik preprocessing
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baris Data Awal", len(df_before))
        with col2:
            st.metric("Baris Data Bersih", len(df_after))
        with col3:
            reduction = ((len(df_before) - len(df_after)) / len(df_before)) * 100
            st.metric("Reduksi Data", f"{reduction:.1f}%")
    
    def plot_rfm_distribution(self, rfm_data):
        """Visualisasi distribusi RFM"""
        # Subplot untuk R, F, M
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Distribusi Recency', 'Distribusi Frequency', 'Distribusi Monetary',
                           'Box Plot Recency', 'Box Plot Frequency', 'Box Plot Monetary'),
            vertical_spacing=0.15
        )
        
        # Histograms
        fig.add_trace(
            go.Histogram(x=rfm_data['Recency'], name='Recency', marker_color='#3498db'),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=rfm_data['Frequency'], name='Frequency', marker_color='#2ecc71'),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=rfm_data['Monetary'], name='Monetary', marker_color='#e74c3c'),
            row=1, col=3
        )
        
        # Box plots
        fig.add_trace(
            go.Box(y=rfm_data['Recency'], name='Recency', marker_color='#3498db'),
            row=2, col=1
        )
        fig.add_trace(
            go.Box(y=rfm_data['Frequency'], name='Frequency', marker_color='#2ecc71'),
            row=2, col=2
        )
        fig.add_trace(
            go.Box(y=rfm_data['Monetary'], name='Monetary', marker_color='#e74c3c'),
            row=2, col=3
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap korelasi RFM
        corr_matrix = rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig_corr.update_layout(
            title="Heatmap Korelasi RFM",
            height=400
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    def plot_rfm_segments(self, rfm_data):
        """Visualisasi segmentasi RFM"""
        # Scatter plot 3D RFM
        fig_3d = px.scatter_3d(
            rfm_data,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='RFM_Total',
            size='RFM_Total',
            hover_data=['CustomerID', 'R_Score', 'F_Score', 'M_Score'],
            title='Visualisasi 3D RFM Analysis',
            color_continuous_scale='Viridis'
        )
        
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Recency (hari)',
                yaxis_title='Frequency',
                zaxis_title='Monetary'
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Segment distribution pie chart
        if 'Segment' in rfm_data.columns:
            segment_counts = rfm_data['Segment'].value_counts()
            
            fig_pie = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title='Distribusi Segment Pelanggan',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # RFM Score distribution
                fig_score = px.histogram(
                    rfm_data,
                    x='RFM_Total',
                    nbins=20,
                    title='Distribusi Total RFM Score',
                    color_discrete_sequence=['#3498db']
                )
                st.plotly_chart(fig_score, use_container_width=True)
    
    def plot_top_categories(self, df):
        """Visualisasi kategori produk teratas"""
        # Top categories by revenue
        category_revenue = df.groupby('Product_Category')['Total_Price'].sum().sort_values(ascending=False).head(10)
        category_quantity = df.groupby('Product_Category')['Quantity'].sum().sort_values(ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                x=category_revenue.values,
                y=category_revenue.index,
                orientation='h',
                title='Top 10 Kategori Berdasarkan Revenue',
                color=category_revenue.values,
                color_continuous_scale='Viridis'
            )
            fig1.update_layout(xaxis_title='Total Revenue', yaxis_title='Kategori')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                x=category_quantity.values,
                y=category_quantity.index,
                orientation='h',
                title='Top 10 Kategori Berdasarkan Quantity',
                color=category_quantity.values,
                color_continuous_scale='Plasma'
            )
            fig2.update_layout(xaxis_title='Total Quantity', yaxis_title='Kategori')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Category distribution over time
        if 'Transaction_Date' in df.columns:
            df['Month'] = df['Transaction_Date'].dt.to_period('M').astype(str)
            monthly_category = df.groupby(['Month', 'Product_Category'])['Total_Price'].sum().reset_index()
            
            top_5_categories = df['Product_Category'].value_counts().head(5).index.tolist()
            monthly_top = monthly_category[monthly_category['Product_Category'].isin(top_5_categories)]
            
            fig_time = px.line(
                monthly_top,
                x='Month',
                y='Total_Price',
                color='Product_Category',
                title='Trend Revenue per Kategori (Bulanan)',
                markers=True
            )
            st.plotly_chart(fig_time, use_container_width=True)
    
    def plot_elbow_method(self, customer_data):
        """Visualisasi Elbow Method untuk menentukan jumlah klaster optimal"""
        # Prepare features for clustering
        features = ['R_Score', 'F_Score', 'M_Score']
        X = customer_data[features].fillna(0)
        
        # Calculate WCSS for different k values
        wcss = []
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if len(X) > k:
                labels = kmeans.labels_
                silhouette_scores.append(silhouette_score(X, labels))
            else:
                silhouette_scores.append(0)
        
        # Plot Elbow Method
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Elbow Method', 'Silhouette Score'),
            horizontal_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(x=list(k_range), y=wcss, mode='lines+markers', name='WCSS',
                      line=dict(color='#e74c3c', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers', name='Silhouette',
                      line=dict(color='#3498db', width=3)),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Jumlah Klaster", row=1, col=1)
        fig.update_xaxes(title_text="Jumlah Klaster", row=1, col=2)
        fig.update_yaxes(title_text="WCSS", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        best_k = k_range[np.argmax(silhouette_scores)]
        st.info(f"💡 **Rekomendasi:** Jumlah klaster optimal berdasarkan Silhouette Score adalah **{best_k}**")
    
    def plot_cluster_3d(self, customer_data):
        """Visualisasi 3D klaster"""
        if 'Cluster' not in customer_data.columns:
            return
        
        fig = px.scatter_3d(
            customer_data,
            x='R_Score',
            y='F_Score',
            z='M_Score',
            color='Cluster',
            size='RFM_Total',
            hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary'],
            title='Visualisasi 3D Klaster Pelanggan',
            color_continuous_scale='Viridis',
            opacity=0.8
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='R Score',
                yaxis_title='F Score',
                zaxis_title='M Score'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_cluster_distribution(self, customer_data):
        """Visualisasi distribusi klaster"""
        if 'Cluster' not in customer_data.columns:
            return
        
        cluster_counts = customer_data['Cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=cluster_counts.values,
                names=cluster_counts.index.astype(str),
                title='Distribusi Pelanggan per Klaster',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                title='Jumlah Pelanggan per Klaster',
                color=cluster_counts.index.astype(str),
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_bar.update_layout(xaxis_title='Klaster', yaxis_title='Jumlah Pelanggan')
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def plot_cluster_profile(self, cluster_profile):
        """Visualisasi profil klaster"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rata-rata R Score per Klaster', 
                          'Rata-rata F Score per Klaster',
                          'Rata-rata M Score per Klaster',
                          'Distribusi Jumlah Pelanggan'),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # R Score
        fig.add_trace(
            go.Bar(x=cluster_profile.index, y=cluster_profile['Avg_R_Score'],
                  name='R Score', marker_color='#3498db'),
            row=1, col=1
        )
        
        # F Score
        fig.add_trace(
            go.Bar(x=cluster_profile.index, y=cluster_profile['Avg_F_Score'],
                  name='F Score', marker_color='#2ecc71'),
            row=1, col=2
        )
        
        # M Score
        fig.add_trace(
            go.Bar(x=cluster_profile.index, y=cluster_profile['Avg_M_Score'],
                  name='M Score', marker_color='#e74c3c'),
            row=2, col=1
        )
        
        # Jumlah Pelanggan
        fig.add_trace(
            go.Bar(x=cluster_profile.index, y=cluster_profile['Jumlah_Pelanggan'],
                  name='Jumlah Pelanggan', marker_color='#9b59b6'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap karakteristik klaster
        heatmap_data = cluster_profile[['Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score']].T
        
        fig_heat = px.imshow(
            heatmap_data,
            labels=dict(x="Klaster", y="Fitur", color="Nilai"),
            title='Heatmap Karakteristik Klaster',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    
    def plot_apriori_results(self, apriori_results):
        """Visualisasi hasil analisis Apriori"""
        if not apriori_results:
            st.warning("Tidak ada hasil analisis Apriori untuk divisualisasikan")
            return
        
        # Collect all rules
        all_rules = []
        for cluster_num, rules in apriori_results.items():
            if len(rules) > 0:
                rules_copy = rules.copy()
                rules_copy['Cluster'] = cluster_num
                all_rules.append(rules_copy.head(5))  # Take top 5 rules per cluster
        
        if not all_rules:
            return
        
        combined_rules = pd.concat(all_rules, ignore_index=True)
        
        # Format antecedents and consequents for display
        combined_rules['Rule'] = combined_rules.apply(
            lambda row: f"{', '.join(list(row['antecedents']))} → {', '.join(list(row['consequents']))}",
            axis=1
        )
        
        # Plot top rules by lift
        top_rules = combined_rules.nlargest(10, 'lift')
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Top Rules by Support', 'Top Rules by Confidence', 'Top Rules by Lift'),
            horizontal_spacing=0.15
        )
        
        # Support
        fig.add_trace(
            go.Bar(x=top_rules['Rule'], y=top_rules['support'], name='Support',
                  marker_color='#3498db'),
            row=1, col=1
        )
        
        # Confidence
        fig.add_trace(
            go.Bar(x=top_rules['Rule'], y=top_rules['confidence'], name='Confidence',
                  marker_color='#2ecc71'),
            row=1, col=2
        )
        
        # Lift
        fig.add_trace(
            go.Bar(x=top_rules['Rule'], y=top_rules['lift'], name='Lift',
                  marker_color='#e74c3c'),
            row=1, col=3
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            xaxis_tickangle=-45,
            xaxis2_tickangle=-45,
            xaxis3_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot support vs confidence
        fig_scatter = px.scatter(
            combined_rules,
            x='support',
            y='confidence',
            size='lift',
            color='Cluster',
            hover_data=['Rule', 'lift'],
            title='Support vs Confidence dengan Lift sebagai Size',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    def plot_promotion_dashboard(self, customer_data, cluster_profile, promotion_strategies):
        """Dashboard visualisasi strategi promosi"""
        if 'Cluster' not in customer_data.columns:
            return
        
        # 1. Customer Value Distribution per Cluster
        fig1 = px.box(
            customer_data,
            x='Cluster',
            y='Monetary',
            color='Cluster',
            title='Distribusi Nilai Pelanggan per Klaster',
            points='all'
        )
        
        # 2. RFM Score Comparison
        fig2 = go.Figure()
        
        clusters = cluster_profile.index.tolist()
        
        fig2.add_trace(go.Bar(
            name='R Score',
            x=clusters,
            y=cluster_profile['Avg_R_Score'],
            marker_color='#3498db'
        ))
        fig2.add_trace(go.Bar(
            name='F Score',
            x=clusters,
            y=cluster_profile['Avg_F_Score'],
            marker_color='#2ecc71'
        ))
        fig2.add_trace(go.Bar(
            name='M Score',
            x=clusters,
            y=cluster_profile['Avg_M_Score'],
            marker_color='#e74c3c'
        ))
        
        fig2.update_layout(
            barmode='group',
            title='Rata-rata RFM Score per Klaster',
            xaxis_title='Klaster',
            yaxis_title='Score'
        )
        
        # 3. Customer Count and Percentage
        fig3 = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Jumlah Pelanggan', 'Persentase Pelanggan'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        fig3.add_trace(
            go.Bar(x=clusters, y=cluster_profile['Jumlah_Pelanggan'],
                  marker_color='#9b59b6', name='Jumlah'),
            row=1, col=1
        )
        
        fig3.add_trace(
            go.Pie(labels=clusters, values=cluster_profile['Jumlah_Pelanggan'],
                  name='Persentase', hole=0.4),
            row=1, col=2
        )
        
        fig3.update_layout(height=400, showlegend=False)
        
        # Display all charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # 4. Strategy Effectiveness Matrix
        st.write("#### 🎯 Matriks Efektivitas Strategi")
        
        # Create effectiveness matrix (example)
        strategies_df = pd.DataFrame({
            'Klaster': clusters,
            'Tipe': ['Premium', 'Loyal', 'Reguler', 'Berisiko'][:len(clusters)],
            'Prioritas': [1, 2, 3, 4][:len(clusters)],
            'Budget_Allocation (%)': [40, 30, 20, 10][:len(clusters)],
            'Expected_ROI (%)': [25, 20, 15, 10][:len(clusters)]
        })
        
        st.dataframe(strategies_df.style.background_gradient(cmap='Blues'), use_container_width=True)