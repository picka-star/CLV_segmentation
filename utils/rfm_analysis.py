import pandas as pd
import numpy as np

class RFMAnalyzer:
    """
    Kelas untuk melakukan analisis RFM sesuai metodologi penelitian
    Subbab 3.8.2: Perhitungan Nilai RFM
    Subbab 3.8.3: Pembentukan Skor RFM
    Subbab 3.8.4: Pembentukan Variabel Kategori Produk
    Subbab 3.8.5: Penyusunan Dataset Pelanggan
    """
    
    def __init__(self, df, reference_date):
        self.df = df.copy()
        self.reference_date = reference_date
        self.rfm_data = None
        self.rfm_scored = None
        
    def calculate_rfm(self):
        """
        Perhitungan nilai RFM (Subbab 3.8.2)
        
        Recency (R) = reference_date - tanggal_transaksi_terakhir_pelanggan (Persamaan 3.2)
        Frequency (F) = jumlah_transaksi_pelanggan (Persamaan 3.3)
        Monetary (M) = Σ TotalPrice_pelanggan (Persamaan 3.5)
        
        Tahap: Customer-level aggregation
        """
        print("\n" + "="*70)
        print("PERHITUNGAN NILAI RFM (Subbab 3.8.2)")
        print("="*70 + "\n")
        
        # Agregasi ke level pelanggan
        self.rfm_data = self.df.groupby('CustomerID').agg({
            'Transaction_Date': lambda x: (self.reference_date - x.max()).days,  # Recency (Persamaan 3.2)
            'Transaction_ID': 'nunique',  # Frequency (Persamaan 3.3)
            'TotalPrice': 'sum'  # Monetary (Persamaan 3.5)
        }).reset_index()
        
        # Rename kolom
        self.rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Validasi: pastikan tidak ada nilai negatif atau tidak wajar
        assert (self.rfm_data['Recency'] >= 0).all(), "❌ Error: Recency negatif terdeteksi"
        assert (self.rfm_data['Frequency'] > 0).all(), "❌ Error: Frequency nol terdeteksi"
        assert (self.rfm_data['Monetary'] > 0).all(), "❌ Error: Monetary tidak konsisten"
        
        print(f"✓ RFM dihitung untuk {len(self.rfm_data):,} pelanggan")
        print(f"\nStatistik RFM:")
        print(f"{'Metrik':<12} {'Min':<12} {'Mean':<12} {'Median':<12} {'Max':<12}")
        print("-" * 60)
        
        for col in ['Recency', 'Frequency', 'Monetary']:
            stats = self.rfm_data[col].describe()
            if col == 'Monetary':
                print(f"{col:<12} ${stats['min']:<11,.0f} ${stats['mean']:<11,.0f} ${stats['50%']:<11,.0f} ${stats['max']:<11,.0f}")
            else:
                print(f"{col:<12} {stats['min']:<12.0f} {stats['mean']:<12.1f} {stats['50%']:<12.0f} {stats['max']:<12.0f}")
        
        return self
    
    def create_rfm_scores(self):
        """
        Pembentukan Skor RFM (Subbab 3.8.3)
        Menggunakan pembagian quintile (skala 1-5)
        
        - R_score: descending (Recency kecil → skor tinggi)
        - F_score: ascending (Frequency besar → skor tinggi)
        - M_score: ascending (Monetary besar → skor tinggi)
        """
        print("\n" + "="*70)
        print("PEMBENTUKAN SKOR RFM (Subbab 3.8.3)")
        print("="*70 + "\n")
        
        self.rfm_scored = self.rfm_data.copy()
        
        # R_score: Recency kecil = skor tinggi (terbalik/descending)
        # Pelanggan dengan transaksi terakhir paling baru mendapat skor tertinggi
        try:
            self.rfm_scored['R_score'] = pd.qcut(
                self.rfm_scored['Recency'], 
                q=5, 
                labels=[5, 4, 3, 2, 1],  # Terbalik: Recency rendah = skor tinggi
                duplicates='drop'
            ).astype(int)
        except ValueError:
            # Jika distribusi tidak memungkinkan qcut, gunakan cut
            self.rfm_scored['R_score'] = pd.cut(
                self.rfm_scored['Recency'], 
                bins=5, 
                labels=[5, 4, 3, 2, 1]
            ).astype(int)
        
        # F_score: Frequency besar = skor tinggi (ascending)
        try:
            self.rfm_scored['F_score'] = pd.qcut(
                self.rfm_scored['Frequency'], 
                q=5, 
                labels=[1, 2, 3, 4, 5],
                duplicates='drop'
            ).astype(int)
        except ValueError:
            self.rfm_scored['F_score'] = pd.cut(
                self.rfm_scored['Frequency'], 
                bins=5, 
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
        
        # M_score: Monetary besar = skor tinggi (ascending)
        try:
            self.rfm_scored['M_score'] = pd.qcut(
                self.rfm_scored['Monetary'], 
                q=5, 
                labels=[1, 2, 3, 4, 5],
                duplicates='drop'
            ).astype(int)
        except ValueError:
            self.rfm_scored['M_score'] = pd.cut(
                self.rfm_scored['Monetary'], 
                bins=5, 
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
        
        # RFM Score gabungan (untuk segmentasi umum)
        self.rfm_scored['RFM_Score'] = (
            self.rfm_scored['R_score'].astype(str) +
            self.rfm_scored['F_score'].astype(str) +
            self.rfm_scored['M_score'].astype(str)
        )
        
        print("✓ Skor RFM dibuat menggunakan pembagian quintile (skala 1-5)")
        print("\nDistribusi Skor:")
        print(f"{'Skor':<10} {'R_score':<12} {'F_score':<12} {'M_score':<12}")
        print("-" * 46)
        for score in [1, 2, 3, 4, 5]:
            r_count = (self.rfm_scored['R_score'] == score).sum()
            f_count = (self.rfm_scored['F_score'] == score).sum()
            m_count = (self.rfm_scored['M_score'] == score).sum()
            print(f"{score:<10} {r_count:<12} {f_count:<12} {m_count:<12}")
        
        # Identifikasi segmen pelanggan berdasarkan RFM
        self.rfm_scored['Customer_Segment'] = self.rfm_scored.apply(self._assign_segment, axis=1)
        
        segment_dist = self.rfm_scored['Customer_Segment'].value_counts()
        print(f"\nSegmentasi Pelanggan Awal:")
        for segment, count in segment_dist.items():
            print(f"  {segment:<20}: {count:>5} pelanggan ({count/len(self.rfm_scored)*100:.1f}%)")
        
        return self
    
    def _assign_segment(self, row):
        """
        Menentukan segmen pelanggan berdasarkan skor RFM
        """
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        
        # Champions: R=5, F=5, M=5 atau kombinasi tinggi
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        # Loyal Customers: F tinggi, R dan M sedang-tinggi
        elif f >= 4 and r >= 3 and m >= 3:
            return 'Loyal Customers'
        # Potential Loyalists: R tinggi, F dan M sedang
        elif r >= 4 and f >= 2 and m >= 2:
            return 'Potential Loyalists'
        # At Risk: F tinggi tapi R rendah
        elif f >= 3 and r <= 2:
            return 'At Risk'
        # Can't Lose Them: F dan M tinggi tapi R rendah
        elif f >= 4 and m >= 4 and r <= 2:
            return 'Cant Lose'
        # Hibernating: R, F, M rendah
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        else:
            return 'Need Attention'
    
    def create_category_proportions(self):
        """
        Pembentukan Variabel Kategori Produk (Subbab 3.8.4)
        
        proporsi_kategori(i) = jumlah_pembelian_kategori(i) / total_pembelian_pelanggan
        (Persamaan 3.6)
        """
        print("\n" + "="*70)
        print("PEMBENTUKAN VARIABEL KATEGORI PRODUK (Subbab 3.8.4)")
        print("="*70 + "\n")
        
        # Hitung jumlah item yang dibeli per kategori per pelanggan
        category_counts = self.df.groupby(['CustomerID', 'Product_Category'])['Quantity'].sum().unstack(fill_value=0)
        
        # Hitung total pembelian per pelanggan (denominator untuk Persamaan 3.6)
        total_purchases = category_counts.sum(axis=1)
        
        # Hitung proporsi menggunakan Persamaan 3.6
        category_proportions = category_counts.div(total_purchases, axis=0)
        
        # Rename kolom dengan prefix 'prop_'
        category_proportions.columns = ['prop_' + col for col in category_proportions.columns]
        
        # Reset index
        category_proportions = category_proportions.reset_index()
        
        print(f"✓ Proporsi kategori dibuat untuk {len(category_proportions.columns)-1} kategori")
        print(f"\nKategori yang ditemukan:")
        categories = [col.replace('prop_', '') for col in category_proportions.columns if col.startswith('prop_')]
        for i, cat in enumerate(sorted(categories), 1):
            print(f"  {i:2}. {cat}")
        
        return category_proportions
    
    def assemble_customer_dataset(self):
        """
        Penyusunan Dataset Pelanggan (Subbab 3.8.5)
        Menggabungkan:
        1. Nilai RFM (Recency, Frequency, Monetary)
        2. Skor RFM (R_score, F_score, M_score)
        3. Proporsi kategori produk
        """
        print("\n" + "="*70)
        print("PENYUSUNAN DATASET PELANGGAN (Subbab 3.8.5)")
        print("="*70 + "\n")
        
        # Dapatkan proporsi kategori
        category_props = self.create_category_proportions()
        
        # Gabungkan dengan skor RFM
        customer_dataset = self.rfm_scored.merge(category_props, on='CustomerID', how='left')
        
        # Isi nilai NaN dengan 0 (pelanggan yang tidak pernah beli kategori tertentu)
        prop_cols = [col for col in customer_dataset.columns if col.startswith('prop_')]
        customer_dataset[prop_cols] = customer_dataset[prop_cols].fillna(0)
        
        # Verifikasi: proporsi harus berjumlah 1 untuk setiap pelanggan
        total_proportions = customer_dataset[prop_cols].sum(axis=1)
        assert np.allclose(total_proportions, 1.0, atol=0.01), "❌ Error: Proporsi tidak berjumlah 1"
        
        print(f"✓ Dataset pelanggan disusun:")
        print(f"  - Total pelanggan: {len(customer_dataset):,}")
        print(f"  - Total kolom: {len(customer_dataset.columns)}")
        print(f"  - Fitur RFM: Recency, Frequency, Monetary")
        print(f"  - Skor RFM: R_score, F_score, M_score")
        print(f"  - Proporsi kategori: {len(prop_cols)} variabel")
        
        print(f"\nStruktur Dataset:")
        print(f"  CustomerID           : ID pelanggan")
        print(f"  Recency, Frequency, Monetary : Nilai RFM mentah")
        print(f"  R_score, F_score, M_score    : Skor RFM (1-5)")
        print(f"  RFM_Score            : Gabungan skor (mis: '555')")
        print(f"  Customer_Segment     : Segmen awal")
        print(f"  prop_*               : {len(prop_cols)} variabel proporsi kategori")
        
        return customer_dataset
    
    def get_rfm_data(self):
        """Mengembalikan data RFM mentah"""
        return self.rfm_data
    
    def get_scored_data(self):
        """Mengembalikan data RFM dengan skor"""
        return self.rfm_scored
    
    def run_all(self):
        """
        Menjalankan seluruh tahap analisis RFM
        """
        self.calculate_rfm()
        self.create_rfm_scores()
        customer_dataset = self.assemble_customer_dataset()
        
        print("\n" + "="*70)
        print("ANALISIS RFM SELESAI")
        print("="*70 + "\n")
        
        return customer_dataset