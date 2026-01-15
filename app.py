import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from utils import DataPreprocessor, RFMAnalyzer, CustomerClustering, AssociationAnalyzer

# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RFM-KMeans-Apriori Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS - TEMA GELAP PROFESIONAL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #667eea !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b8b8d1 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Cards */
    .custom-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 8px 32px 0 rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.5);
    }
    
    /* DataFrames */
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px;
        color: #ffffff !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #b8b8d1;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Info boxes */
    .element-container div[data-testid="stMarkdownContainer"] > p {
        color: #d1d1e0;
        line-height: 1.6;
    }
    
    /* Success/Info/Warning boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def create_metric_card(label, value, delta=None):
    """Membuat kartu metrik kustom"""
    delta_html = f"<p style='color: #4ade80; font-size: 0.9rem; margin-top: 0.5rem;'>↑ {delta}</p>" if delta else ""
    
    return f"""
    <div class="custom-card">
        <p style="color: #b8b8d1; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
            {label}
        </p>
        <h2 style="color: #667eea; margin: 0; font-size: 2rem; font-weight: 700;">
            {value}
        </h2>
        {delta_html}
    </div>
    """

def plot_theme():
    """Template tema gelap untuk plotly"""
    return {
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(255,255,255,0.03)',
        'font': {'color': '#d1d1e0', 'family': 'Inter'},
        'xaxis': {
            'gridcolor': 'rgba(255,255,255,0.08)',
            'zerolinecolor': 'rgba(255,255,255,0.1)'
        },
        'yaxis': {
            'gridcolor': 'rgba(255,255,255,0.08)',
            'zerolinecolor': 'rgba(255,255,255,0.1)'
        }
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📊 RFM-KMeans-Apriori Analytics")
    st.markdown("---")
    
    st.markdown("#### 📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload file dataset",
        type=['xlsx', 'xls', 'csv'],
        help="Upload dataset transaksi pelanggan (Excel atau CSV)"
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ Parameter Klasterisasi")
    
    n_clusters = st.slider(
        "Jumlah Cluster",
        min_value=2,
        max_value=10,
        value=4,
        help="Jumlah segmen pelanggan yang akan dibentuk"
    )
    
    st.markdown("---")
    st.markdown("#### 🔍 Parameter Apriori")
    
    min_support = st.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Minimum frekuensi kemunculan itemset"
    )
    
    min_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum tingkat kepercayaan rule"
    )
    
    min_lift = st.slider(
        "Minimum Lift",
        min_value=1.0,
        max_value=5.0,
        value=1.2,
        step=0.1,
        help="Minimum kekuatan asosiasi"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; border-left: 3px solid #667eea;">
        <p style="font-size: 0.85rem; color: #b8b8d1; margin: 0;">
            💡 <strong>Metodologi:</strong><br>
            1. Pra-pemrosesan Data<br>
            2. Analisis RFM<br>
            3. K-Means Clustering<br>
            4. Apriori Association Rules<br>
            5. Strategi Promosi
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("# 📊 Customer Segmentation & Market Basket Analysis")
st.markdown("### Analisis Pola Pembelian dengan RFM, K-Means, dan Apriori")

if uploaded_file is None:
    st.markdown("---")
    st.info("👆 Silakan upload file dataset di sidebar untuk memulai analisis")
    
    # Tampilkan contoh struktur data yang diharapkan
    st.markdown("#### 📋 Format Data yang Diharapkan")
    
    sample_data = pd.DataFrame({
        'CustomerID': [17850, 17850, 17850],
        'Transaction_ID': [16679, 16680, 16696],
        'Transaction_Date': ['2019-01-01', '2019-01-01', '2019-01-01'],
        'Product_SKU': ['GGOENEBJ079499', 'GGOENEBJ079499', 'GGOENEBQ078999'],
        'Product_Description': ['Nest Learning Thermostat', 'Nest Learning Thermostat', 'Nest Cam Outdoor'],
        'Product_Category': ['Nest-USA', 'Nest-USA', 'Nest-USA'],
        'Quantity': [1, 1, 2],
        'Avg_Price': [249.0, 249.0, 199.0]
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
else:
    # Load data
    try:
        # Deteksi tipe file dan load
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        
        # Tampilkan info dataset
        st.success(f"✅ Dataset berhasil dimuat: {len(df_raw):,} baris × {len(df_raw.columns)} kolom")
        
        # Tabs untuk navigasi
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Dashboard",
            "🔄 Pra-pemrosesan",
            "💎 Analisis RFM",
            "🎯 Segmentasi",
            "🔗 Pola Asosiasi"
        ])
        
        # ═══════════════════════════════════════════════════════════════
        # TAB 1: DASHBOARD
        # ═══════════════════════════════════════════════════════════════
        
        with tab1:
            st.markdown("## 📊 Dashboard Overview")
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    create_metric_card(
                        "Total Transaksi",
                        f"{len(df_raw):,}"
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    create_metric_card(
                        "Total Pelanggan",
                        f"{df_raw['CustomerID'].nunique():,}"
                    ),
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    create_metric_card(
                        "Kategori Produk",
                        f"{df_raw['Product_Category'].nunique()}"
                    ),
                    unsafe_allow_html=True
                )
            
            with col4:
                total_revenue = (df_raw['Quantity'] * df_raw['Avg_Price']).sum()
                st.markdown(
                    create_metric_card(
                        "Total Revenue",
                        f"${total_revenue:,.0f}"
                    ),
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # Visualisasi overview
            col1, col2 = st.columns(2)
            
            with col1:
                # Top Categories
                cat_sales = df_raw.groupby('Product_Category').agg({
                    'Quantity': 'sum'
                }).nlargest(10, 'Quantity').reset_index()
                
                fig = px.bar(
                    cat_sales,
                    x='Quantity',
                    y='Product_Category',
                    orientation='h',
                    title='Top 10 Kategori Produk',
                    color='Quantity',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(**plot_theme())
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Transaksi per bulan
                df_temp = df_raw.copy()
                df_temp['Month'] = pd.to_datetime(df_temp['Transaction_Date']).dt.to_period('M').astype(str)
                monthly = df_temp.groupby('Month').size().reset_index(name='Transactions')
                
                fig = px.line(
                    monthly,
                    x='Month',
                    y='Transactions',
                    title='Tren Transaksi Bulanan',
                    markers=True
                )
                fig.update_layout(**plot_theme())
                fig.update_traces(line_color='#667eea', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
        
        # ═══════════════════════════════════════════════════════════════
        # TAB 2: PRA-PEMROSESAN
        # ═══════════════════════════════════════════════════════════════
        
        with tab2:
            st.markdown("## 🔄 Pra-pemrosesan Data")
            
            with st.spinner("Memproses data..."):
                # Jalankan preprocessing
                preprocessor = DataPreprocessor(df_raw)
                preprocessor.run_all()
                df_clean = preprocessor.get_processed_data()
                
                st.success("✅ Data berhasil diproses!")
                
                # Tampilkan statistik
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Sebelum Pemrosesan")
                    st.metric("Jumlah Baris", f"{len(df_raw):,}")
                    st.metric("Jumlah Kolom", len(df_raw.columns))
                
                with col2:
                    st.markdown("### Setelah Pemrosesan")
                    st.metric("Jumlah Baris", f"{len(df_clean):,}")
                    st.metric("Jumlah Kolom", len(df_clean.columns))
                
                # Preview data
                st.markdown("### Preview Data Bersih")
                st.dataframe(df_clean.head(20), use_container_width=True)
                
                # Info tentang tahapan
                with st.expander("ℹ️ Tahapan Pra-pemrosesan"):
                    st.markdown("""
                    1. **Pembersihan Data**: Hapus duplikat dan nilai tidak valid
                    2. **Penanganan Missing Values**: Eliminasi baris dengan nilai kosong
                    3. **Seleksi Fitur**: Pertahankan kolom relevan untuk RFM
                    4. **Variabel Turunan**: Hitung TotalPrice = Quantity × Avg_Price
                    5. **Standarisasi Kategori**: Normalisasi nama kategori produk
                    6. **Standarisasi Tipe Data**: Konversi ke tipe yang sesuai
                    """)
        
        # ═══════════════════════════════════════════════════════════════
        # TAB 3: ANALISIS RFM
        # ═══════════════════════════════════════════════════════════════
        
        with tab3:
            st.markdown("## 💎 Analisis RFM")
            
            if 'df_clean' not in locals():
                preprocessor = DataPreprocessor(df_raw)
                preprocessor.run_all()
                df_clean = preprocessor.get_processed_data()
            
            with st.spinner("Menghitung RFM..."):
                # Jalankan RFM analysis
                rfm_analyzer = RFMAnalyzer(df_clean, preprocessor.reference_date)
                customer_dataset = rfm_analyzer.run_all()
                
                st.success("✅ Analisis RFM selesai!")
                
                # Distribusi RFM Scores
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig = px.histogram(
                        customer_dataset,
                        x='R_score',
                        title='Distribusi Recency Score',
                        nbins=5,
                        color_discrete_sequence=['#667eea']
                    )
                    fig.update_layout(**plot_theme())
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        customer_dataset,
                        x='F_score',
                        title='Distribusi Frequency Score',
                        nbins=5,
                        color_discrete_sequence=['#764ba2']
                    )
                    fig.update_layout(**plot_theme())
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    fig = px.histogram(
                        customer_dataset,
                        x='M_score',
                        title='Distribusi Monetary Score',
                        nbins=5,
                        color_discrete_sequence=['#f093fb']
                    )
                    fig.update_layout(**plot_theme())
                    st.plotly_chart(fig, use_container_width=True)
                
                # 3D Scatter RFM
                st.markdown("### Visualisasi 3D RFM")
                fig = px.scatter_3d(
                    customer_dataset.sample(min(1000, len(customer_dataset))),
                    x='R_score',
                    y='F_score',
                    z='M_score',
                    color='RFM_Score',
                    title='Pemetaan 3D Customer RFM',
                    opacity=0.7
                )
                fig.update_layout(**plot_theme(), height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabel RFM
                st.markdown("### Data RFM Pelanggan")
                st.dataframe(
                    customer_dataset[['CustomerID', 'Recency', 'Frequency', 'Monetary', 
                                     'R_score', 'F_score', 'M_score', 'RFM_Score']].head(50),
                    use_container_width=True
                )
        
        # ═══════════════════════════════════════════════════════════════
        # TAB 4: SEGMENTASI K-MEANS
        # ═══════════════════════════════════════════════════════════════
        
        with tab4:
            st.markdown("## 🎯 Segmentasi Pelanggan dengan K-Means")
            
            if 'customer_dataset' not in locals():
                preprocessor = DataPreprocessor(df_raw)
                preprocessor.run_all()
                df_clean = preprocessor.get_processed_data()
                rfm_analyzer = RFMAnalyzer(df_clean, preprocessor.reference_date)
                customer_dataset = rfm_analyzer.run_all()
            
            with st.spinner("Melakukan klasterisasi..."):
                # Jalankan clustering
                clustering = CustomerClustering(customer_dataset)
                results = clustering.run_all(n_clusters=n_clusters)
                
                clustered_data = results['clustered_data']
                elbow_data = results['elbow_data']
                profiles = results['profiles']
                
                st.success(f"✅ Segmentasi selesai! {n_clusters} cluster terbentuk")
                
                # Elbow Method & Silhouette
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=elbow_data['k_values'],
                        y=elbow_data['wcss'],
                        mode='lines+markers',
                        name='WCSS',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=10)
                    ))
                    fig.update_layout(
                        title='Elbow Method',
                        xaxis_title='Jumlah Cluster',
                        yaxis_title='WCSS',
                        **plot_theme()
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=elbow_data['k_values'],
                        y=elbow_data['silhouette_scores'],
                        mode='lines+markers',
                        name='Silhouette',
                        line=dict(color='#764ba2', width=3),
                        marker=dict(size=10)
                    ))
                    fig.update_layout(
                        title='Silhouette Score',
                        xaxis_title='Jumlah Cluster',
                        yaxis_title='Silhouette Score',
                        **plot_theme()
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Distribusi Cluster
                st.markdown("### Distribusi Pelanggan per Cluster")
                cluster_dist = clustered_data['Cluster'].value_counts().reset_index()
                cluster_dist.columns = ['Cluster', 'Count']
                
                fig = px.pie(
                    cluster_dist,
                    values='Count',
                    names='Cluster',
                    title='Proporsi Pelanggan per Cluster',
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                fig.update_layout(**plot_theme())
                st.plotly_chart(fig, use_container_width=True)
                
                # Profil Cluster
                st.markdown("### Profil Karakteristik Cluster")
                st.dataframe(profiles['rfm_profile'], use_container_width=True)
                
                # Top Kategori per Cluster
                st.markdown("### Top 3 Kategori Produk per Cluster")
                for cluster_id, categories in profiles['top_categories'].items():
                    with st.expander(f"📦 Cluster {cluster_id}"):
                        for i, (cat, pct) in enumerate(categories, 1):
                            st.markdown(f"**{i}.** {cat.replace('_', ' ').title()} — {pct}")
                
                # 3D Visualization
                st.markdown("### Visualisasi 3D Cluster")
                fig = px.scatter_3d(
                    clustered_data.sample(min(1000, len(clustered_data))),
                    x='R_score',
                    y='F_score',
                    z='M_score',
                    color='Cluster',
                    title='Pemetaan 3D Customer Clusters',
                    opacity=0.7,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(**plot_theme(), height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        # ═══════════════════════════════════════════════════════════════
        # TAB 5: POLA ASOSIASI
        # ═══════════════════════════════════════════════════════════════
        
        with tab5:
            st.markdown("## 🔗 Analisis Pola Asosiasi dengan Apriori")
            
            if 'clustered_data' not in locals():
                preprocessor = DataPreprocessor(df_raw)
                preprocessor.run_all()
                df_clean = preprocessor.get_processed_data()
                rfm_analyzer = RFMAnalyzer(df_clean, preprocessor.reference_date)
                customer_dataset = rfm_analyzer.run_all()
                clustering = CustomerClustering(customer_dataset)
                results = clustering.run_all(n_clusters=n_clusters)
                clustered_data = results['clustered_data']
            
            with st.spinner("Menganalisis pola asosiasi..."):
                # Jalankan Apriori
                apriori_analyzer = AssociationAnalyzer(df_clean, clustered_data)
                all_rules = apriori_analyzer.analyze_all_clusters(
                    min_support=min_support,
                    min_confidence=min_confidence,
                    min_lift=min_lift
                )
                
                if all_rules is not None and len(all_rules) > 0:
                    st.success(f"✅ {len(all_rules)} association rules ditemukan!")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(
                            create_metric_card(
                                "Total Rules",
                                f"{len(all_rules)}"
                            ),
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        avg_confidence = all_rules['confidence'].mean()
                        st.markdown(
                            create_metric_card(
                                "Avg Confidence",
                                f"{avg_confidence:.2%}"
                            ),
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        avg_lift = all_rules['lift'].mean()
                        st.markdown(
                            create_metric_card(
                                "Avg Lift",
                                f"{avg_lift:.2f}"
                            ),
                            unsafe_allow_html=True
                        )
                    
                    # Top Rules
                    st.markdown("### 🏆 Top 20 Association Rules (berdasarkan Lift)")
                    top_rules = all_rules.nlargest(20, 'lift')[
                        ['Cluster', 'antecedents_str', 'consequents_str', 
                         'support', 'confidence', 'lift']
                    ]
                    st.dataframe(top_rules, use_container_width=True)
                    
                    # Visualisasi per Cluster
                    st.markdown("### 📊 Rules per Cluster")
                    
                    cluster_rule_count = all_rules.groupby('Cluster').size().reset_index(name='Rule Count')
                    
                    fig = px.bar(
                        cluster_rule_count,
                        x='Cluster',
                        y='Rule Count',
                        title='Jumlah Rules per Cluster',
                        color='Rule Count',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(**plot_theme())
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Scatter: Confidence vs Lift
                    st.markdown("### 🎯 Confidence vs Lift")
                    fig = px.scatter(
                        all_rules,
                        x='confidence',
                        y='lift',
                        color='Cluster',
                        size='support',
                        hover_data=['antecedents_str', 'consequents_str'],
                        title='Association Rules Quality',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(**plot_theme())
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Rekomendasi Strategi
                    st.markdown("### 💡 Rekomendasi Strategi Promosi")
                    
                    recommendations = apriori_analyzer.get_all_recommendations()
                    
                    for rec in recommendations:
                        with st.expander(f"🎯 Cluster {rec['cluster_id']} — {rec['customer_count']} pelanggan"):
                            st.markdown(f"**Profil RFM:** {rec['avg_rfm_score']}")
                            
                            for strategy in rec['strategies'][:6]:  # Top 6 strategies
                                st.markdown(f"""
                                <div class="custom-card">
                                    <strong style="color: #667eea;">{strategy['type']}</strong><br>
                                    {strategy['description']}<br>
                                    <small style="color: #b8b8d1;">{strategy['metrics']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Download rules
                    st.markdown("### 💾 Download Association Rules")
                    csv = all_rules.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="association_rules.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("⚠️ Tidak ada association rules yang ditemukan dengan parameter saat ini. Coba turunkan threshold minimum support atau confidence.")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #b8b8d1; padding: 2rem;">
    <p style="margin: 0;">RFM-KMeans-Apriori Analytics Platform</p>
    <p style="font-size: 0.85rem; margin-top: 0.5rem;">
        Analisis Pola Pembelian untuk Strategi Promosi Berbasis Data
    </p>
</div>
""", unsafe_allow_html=True)