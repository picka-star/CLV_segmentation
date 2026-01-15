import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import modul custom
from modules.preprocessing import DataPreprocessor
from modules.rfm import RFMAnalyzer
from modules.clustering import CustomerClustering
from modules.apriori import AprioriAnalyzer
from modules.promotion import PromotionStrategy
from modules.visualization import DataVisualizer

st.set_page_config(
    page_title="Analisis Segmentasi Pelanggan Online Store",
    page_icon="📊",
    layout="wide"
)

class MainApp:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.rfm_analyzer = RFMAnalyzer()
        self.clustering = CustomerClustering()
        self.apriori = AprioriAnalyzer()
        self.promotion = PromotionStrategy()
        self.visualizer = DataVisualizer()
        
    def run(self):
        # Header
        st.title("🎓 Analisis Pola Pembelian Produk pada Online Store")
        st.markdown("""
        **Implementasi Skripsi:** Analisis menggunakan RFM, K-Means, dan Apriori 
        untuk Menyusun Strategi Promosi Berdasarkan Segmentasi Pelanggan
        """)
        
        # Sidebar
        st.sidebar.header("⚙️ Konfigurasi Analisis")
        
        # Upload data
        uploaded_file = st.sidebar.file_uploader(
            "📁 Upload Dataset",
            type=['csv', 'xlsx', 'xls'],
            help="Upload dataset transaksi online store"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.sidebar.success(f"✅ Dataset berhasil diunggah")
                st.sidebar.write(f"**Ukuran Data:** {df.shape[0]} baris × {df.shape[1]} kolom")
                
                with st.sidebar.expander("👁️ Preview Data"):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Main analysis process
                if st.button("🚀 Jalankan Analisis Lengkap", type="primary", use_container_width=True):
                    self.run_complete_analysis(df)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            self.show_instructions()
    
    def run_complete_analysis(self, df):
        """Run complete analysis pipeline with visualizations"""
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ========== Step 1: Preprocessing ==========
        with st.expander("🧹 Tahap 1: Pra-Pemrosesan Data", expanded=True):
            status_text.text("1/6: Memproses data...")
            df_clean = self.preprocessor.preprocess(df)
            progress_bar.progress(16)
            
            # Visualisasi data preprocessing
            st.write("#### 📊 Visualisasi Data Sebelum dan Sesudah Preprocessing")
            self.visualizer.plot_preprocessing_comparison(df, df_clean)
        
        # ========== Step 2: RFM Analysis ==========
        with st.expander("📈 Tahap 2: Analisis RFM", expanded=True):
            status_text.text("2/6: Menghitung RFM...")
            rfm_data, rfm_summary = self.rfm_analyzer.calculate_rfm(df_clean)
            progress_bar.progress(32)
            
            # Visualisasi RFM
            st.write("#### 📊 Visualisasi Distribusi RFM")
            self.visualizer.plot_rfm_distribution(rfm_data)
            
            st.write("#### 🎯 Segmentasi RFM")
            self.visualizer.plot_rfm_segments(rfm_data)
            
            # Analisis kategori produk
            status_text.text("Analisis kategori produk...")
            customer_data = self.rfm_analyzer.analyze_product_categories(df_clean, rfm_data)
            
            # Visualisasi kategori produk
            st.write("#### 🏷️ Top Kategori Produk")
            self.visualizer.plot_top_categories(df_clean)
        
        # ========== Step 3: Clustering ==========
        with st.expander("🔮 Tahap 3: Klasterisasi K-Means", expanded=True):
            status_text.text("3/6: Melakukan klasterisasi...")
            
            # Tentukan jumlah klaster optimal
            st.write("#### 🔍 Menentukan Jumlah Klaster Optimal")
            self.visualizer.plot_elbow_method(customer_data)
            
            # Parameter clustering
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Pilih jumlah klaster:", 2, 8, 4)
            with col2:
                use_category = st.checkbox("Gunakan kategori produk dalam clustering", value=True)
            
            # Lakukan clustering
            customer_data, cluster_profile = self.clustering.perform_clustering(
                customer_data, n_clusters=n_clusters, use_category=use_category
            )
            progress_bar.progress(48)
            
            # Visualisasi clustering
            st.write("#### 📊 Visualisasi Klaster")
            
            col1, col2 = st.columns(2)
            with col1:
                self.visualizer.plot_cluster_3d(customer_data)
            with col2:
                self.visualizer.plot_cluster_distribution(customer_data)
            
            st.write("#### 📈 Profil Klaster")
            self.visualizer.plot_cluster_profile(cluster_profile)
            
            # Tabel profil klaster
            st.dataframe(cluster_profile, use_container_width=True)
        
        # ========== Step 4: Apriori Analysis ==========
        with st.expander("🔗 Tahap 4: Analisis Pola Asosiasi (Apriori)", expanded=True):
            status_text.text("4/6: Analisis pola asosiasi...")
            
            # Parameter Apriori
            st.write("#### ⚙️ Parameter Analisis Apriori")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_support = st.slider("Minimum Support:", 0.01, 0.3, 0.05, 0.01)
            with col2:
                min_confidence = st.slider("Minimum Confidence:", 0.1, 0.9, 0.5, 0.05)
            with col3:
                min_lift = st.slider("Minimum Lift:", 1.0, 3.0, 1.2, 0.1)
            
            # Lakukan analisis Apriori
            apriori_results = self.apriori.analyze(
                df_clean, customer_data, 
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift
            )
            progress_bar.progress(64)
            
            # Visualisasi Apriori
            if apriori_results:
                st.write("#### 📊 Pola Asosiasi per Klaster")
                self.visualizer.plot_apriori_results(apriori_results)
            else:
                st.warning("Tidak ditemukan pola asosiasi yang signifikan")
        
        # ========== Step 5: Promotion Strategies ==========
        with st.expander("🎯 Tahap 5: Strategi Promosi", expanded=True):
            status_text.text("5/6: Menyusun strategi promosi...")
            
            # Generate strategies
            promotion_strategies = self.promotion.generate_strategies(
                customer_data, cluster_profile, apriori_results
            )
            progress_bar.progress(80)
            
            # Visualisasi strategi
            st.write("#### 📊 Dashboard Strategi Promosi")
            self.visualizer.plot_promotion_dashboard(customer_data, cluster_profile, promotion_strategies)
        
        # ========== Step 6: Export Results ==========
        with st.expander("💾 Tahap 6: Ekspor Hasil", expanded=True):
            status_text.text("6/6: Menyiapkan hasil untuk diunduh...")
            self.export_results(customer_data, cluster_profile, apriori_results)
            progress_bar.progress(100)
            status_text.text("✅ Analisis selesai!")
            
            st.balloons()
            st.success("🎉 Analisis berhasil diselesaikan! Semua tahapan telah dieksekusi.")
    
    def show_instructions(self):
        """Show instructions and sample data"""
        st.info("👈 Silakan unggah dataset melalui sidebar")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### 📝 Contoh Format Dataset")
            example_data = pd.DataFrame({
                'CustomerID': [17850, 17850, 13047, 12583, 15100],
                'Transaction_ID': [16679, 16680, 16684, 16694, 16712],
                'Transaction_Date': ['2019-01-01', '2019-01-01', '2019-01-01', '2019-01-01', '2019-01-01'],
                'Product_SKU': ['GGOENEBJ079499', 'GGOENEBJ079499', 'GGOENEBQ078999', 'GGOENEBB078899', 'GGOENEBJ079499'],
                'Product_Description': ['Nest Learning Thermostat', 'Nest Learning Thermostat', 
                                       'Nest Cam Outdoor', 'Nest Cam Indoor', 'Nest Learning Thermostat'],
                'Product_Category': ['Nest-USA', 'Nest-USA', 'Nest-USA', 'Nest-USA', 'Nest-USA'],
                'Quantity': [1, 1, 2, 1, 1],
                'Avg_Price': [153.71, 153.71, 122.77, 122.77, 153.71]
            })
            st.dataframe(example_data, use_container_width=True)
        
        with col2:
            st.write("### 🎯 Kolom Wajib")
            st.write("""
            - **CustomerID**: ID pelanggan
            - **Transaction_ID**: ID transaksi
            - **Transaction_Date**: Tanggal transaksi
            - **Product_Category**: Kategori produk
            - **Quantity**: Jumlah barang
            - **Avg_Price**: Harga satuan
            """)
            
            st.write("### 📊 Metode Analisis")
            st.write("""
            1. RFM Analysis
            2. K-Means Clustering
            3. Apriori Algorithm
            4. Promotion Strategy
            """)
    
    def export_results(self, customer_data, cluster_profile, apriori_results):
        """Export analysis results"""
        st.write("### 📤 Ekspor Hasil Analisis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_rfm = customer_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Data RFM & Cluster",
                data=csv_rfm,
                file_name="rfm_cluster_analysis.csv",
                mime="text/csv",
                help="Data lengkap RFM, skor, segment, dan cluster"
            )
        
        with col2:
            csv_summary = cluster_profile.to_csv()
            st.download_button(
                label="📊 Download Summary Cluster",
                data=csv_summary,
                file_name="cluster_summary.csv",
                mime="text/csv",
                help="Statistik dan profil setiap cluster"
            )
        
        with col3:
            report = self.generate_report(customer_data, cluster_profile, apriori_results)
            st.download_button(
                label="📄 Download Laporan Lengkap",
                data=report,
                file_name="analisis_segmentasi_laporan.txt",
                mime="text/plain",
                help="Laporan analisis lengkap dalam format text"
            )
    
    def generate_report(self, customer_data, cluster_profile, apriori_results):
        """Generate text report"""
        report = "=" * 60 + "\n"
        report += "LAPORAN ANALISIS SEGMENTASI PELANGGAN\n"
        report += "=" * 60 + "\n\n"
        
        # 1. Ringkasan Umum
        report += "1. RINGKASAN UMUM\n"
        report += "-" * 40 + "\n"
        report += f"Total Pelanggan: {len(customer_data)}\n"
        report += f"Jumlah Klaster: {len(cluster_profile)}\n\n"
        
        # 2. Distribusi Klaster
        report += "2. DISTRIBUSI KLASTER\n"
        report += "-" * 40 + "\n"
        for idx, row in cluster_profile.iterrows():
            report += f"\nKlaster {idx}:\n"
            report += f"  - Jumlah Pelanggan: {int(row['Jumlah_Pelanggan'])} ({row['Persentase']}%)\n"
            report += f"  - Rata-rata RFM: R={row['Avg_R_Score']:.1f}, F={row['Avg_F_Score']:.1f}, M={row['Avg_M_Score']:.1f}\n"
        
        # 3. Rekomendasi Strategi
        report += "\n\n3. REKOMENDASI STRATEGI PROMOSI\n"
        report += "-" * 40 + "\n"
        
        for cluster_num in cluster_profile.index:
            report += f"\nKlaster {cluster_num}:\n"
            profile = cluster_profile.loc[cluster_num]
            
            # Tentukan tipe strategi
            if profile['Avg_R_Score'] >= 4 and profile['Avg_F_Score'] >= 4 and profile['Avg_M_Score'] >= 4:
                report += "  Tipe: Pelanggan Premium\n"
                report += "  Strategi: Program loyalty eksklusif\n"
            elif profile['Avg_R_Score'] >= 4 and profile['Avg_F_Score'] >= 3:
                report += "  Tipe: Pelanggan Loyal\n"
                report += "  Strategi: Cross-selling dan bundle discount\n"
            elif profile['Avg_R_Score'] <= 2 and profile['Avg_F_Score'] >= 3:
                report += "  Tipe: Pelanggan Berisiko\n"
                report += "  Strategi: Re-activation campaign\n"
            else:
                report += "  Tipe: Pelanggan Reguler\n"
                report += "  Strategi: Regular promotion\n"
        
        report += "\n\n" + "=" * 60
        return report

if __name__ == "__main__":
    app = MainApp()
    app.run()