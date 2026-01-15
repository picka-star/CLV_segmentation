import pandas as pd
import numpy as np
from datetime import datetime

class DataPreprocessor:
    """
    Kelas untuk melakukan pra-pemrosesan data sesuai metodologi penelitian
    Subbab 3.8.1: Pra-Pemrosesan Data
    
    Dataset: 52,955 transaksi dengan 21 kolom
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.reference_date = None
        self.original_shape = df.shape
        
    def clean_data(self):
        """
        Tahap 1: Pembersihan data
        - Hapus duplikat berdasarkan CustomerID dan Transaction_ID
        - Verifikasi nilai Quantity dan Avg_Price
        - Validasi Transaction_Date
        """
        initial_rows = len(self.df)
        
        # Hapus kolom index yang tidak perlu (kolom unnamed)
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(columns=['Unnamed: 0'])
        if '' in self.df.columns:
            self.df = self.df.drop(columns=[''])
        
        # Konversi CustomerID dan Transaction_ID ke numerik, handle NaN
        self.df['CustomerID'] = pd.to_numeric(self.df['CustomerID'], errors='coerce')
        self.df['Transaction_ID'] = pd.to_numeric(self.df['Transaction_ID'], errors='coerce')
        
        # Hapus baris dengan CustomerID atau Transaction_ID kosong
        self.df = self.df.dropna(subset=['CustomerID', 'Transaction_ID'])
        
        # Konversi ke integer
        self.df['CustomerID'] = self.df['CustomerID'].astype(int)
        self.df['Transaction_ID'] = self.df['Transaction_ID'].astype(int)
        
        # Hapus duplikat berdasarkan CustomerID dan Transaction_ID
        self.df = self.df.drop_duplicates(subset=['CustomerID', 'Transaction_ID'], keep='first')
        
        # Hapus nilai negatif atau nol pada Quantity dan Avg_Price
        self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce')
        self.df['Avg_Price'] = pd.to_numeric(self.df['Avg_Price'], errors='coerce')
        
        self.df = self.df[
            (self.df['Quantity'] > 0) & 
            (self.df['Avg_Price'] > 0)
        ]
        
        # Konversi Transaction_Date ke datetime
        # Format: "2019-01-01 00:00:00" atau "2019-01-01"
        self.df['Transaction_Date'] = pd.to_datetime(self.df['Transaction_Date'], errors='coerce')
        self.df = self.df[self.df['Transaction_Date'].notna()]
        
        cleaned_rows = len(self.df)
        print(f"✓ Pembersihan Data: {initial_rows:,} → {cleaned_rows:,} baris ({initial_rows-cleaned_rows:,} dihapus)")
        
        return self
    
    def handle_missing_values(self):
        """
        Tahap 2: Penanganan nilai hilang
        Eliminasi baris dengan nilai kosong pada atribut kunci
        """
        required_cols = ['CustomerID', 'Transaction_ID', 'Transaction_Date', 
                        'Product_Category', 'Quantity', 'Avg_Price']
        
        initial_rows = len(self.df)
        
        # Hapus baris dengan nilai kosong pada kolom kunci
        self.df = self.df.dropna(subset=required_cols)
        
        # Hapus baris dengan Product_Category kosong atau 'nan' string
        self.df = self.df[self.df['Product_Category'].astype(str).str.strip() != '']
        self.df = self.df[self.df['Product_Category'].astype(str).str.lower() != 'nan']
        
        cleaned_rows = len(self.df)
        print(f"✓ Missing Values: {initial_rows:,} → {cleaned_rows:,} baris ({initial_rows-cleaned_rows:,} dihapus)")
        
        return self
    
    def select_features(self):
        """
        Tahap 3: Seleksi atribut
        Sesuai metodologi: hanya pertahankan kolom yang relevan untuk RFM dan kategori
        
        Kolom yang dipertahankan:
        - CustomerID: ID pelanggan
        - Transaction_ID: ID transaksi
        - Transaction_Date: Tanggal transaksi (untuk Recency)
        - Product_Category: Kategori produk (untuk proporsi)
        - Quantity: Jumlah item (untuk Monetary)
        - Avg_Price: Harga rata-rata (untuk Monetary)
        
        Kolom yang dieliminasi (sesuai batasan masalah):
        - Gender, Location, Tenure_Months: demografi (tidak dianalisis)
        - Delivery_Charges, GST, Discount_pct: biaya tambahan
        - Coupon_Status, Coupon_Code: promosi
        - Offline_Spend, Online_Spend: tidak relevan dengan transaksi
        - Month, Date: redundan dengan Transaction_Date
        - Product_SKU, Product_Description: detail produk (kategori sudah cukup)
        """
        relevant_cols = [
            'CustomerID', 
            'Transaction_ID', 
            'Transaction_Date',
            'Product_Category', 
            'Quantity', 
            'Avg_Price'
        ]
        
        initial_cols = len(self.df.columns)
        self.df = self.df[relevant_cols]
        
        print(f"✓ Seleksi Fitur: {initial_cols} → {len(relevant_cols)} kolom")
        print(f"  Kolom dipertahankan: {', '.join(relevant_cols)}")
        
        return self
    
    def create_derived_variables(self):
        """
        Tahap 4: Pembentukan variabel turunan
        TotalPrice = Quantity × Avg_Price (Persamaan 3.1)
        """
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['Avg_Price']
        
        # Validasi TotalPrice
        invalid_prices = (self.df['TotalPrice'] <= 0).sum()
        if invalid_prices > 0:
            self.df = self.df[self.df['TotalPrice'] > 0]
            print(f"  ⚠ {invalid_prices} baris dengan TotalPrice ≤ 0 dihapus")
        
        print(f"✓ Variabel TotalPrice dibuat: Quantity × Avg_Price")
        print(f"  Total Revenue: ${self.df['TotalPrice'].sum():,.2f}")
        
        return self
    
    def standardize_categories(self):
        """
        Tahap 5: Penyeragaman kategori produk
        - Lowercase
        - Strip whitespace
        - Replace space dengan underscore
        - Konsistensi penamaan
        """
        initial_categories = self.df['Product_Category'].nunique()
        
        # Standarisasi format
        self.df['Product_Category'] = (
            self.df['Product_Category']
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(' ', '_')
            .str.replace('-', '_')
            .str.replace('&', 'and')
        )
        
        # Penyeragaman nama kategori yang mirip
        category_mapping = {
            'nest_usa': 'nest_usa',
            'nest_canada': 'nest_canada',
            'notebooks_and_journals': 'notebooks_journals',
            'more_bags': 'bags_more',
        }
        
        self.df['Product_Category'] = self.df['Product_Category'].replace(category_mapping)
        
        # Hapus kategori yang tidak valid
        self.df = self.df[self.df['Product_Category'] != 'nan']
        self.df = self.df[self.df['Product_Category'] != '']
        
        final_categories = self.df['Product_Category'].nunique()
        
        print(f"✓ Kategori Produk Diseragamkan: {initial_categories} → {final_categories} kategori unik")
        
        # Tampilkan kategori yang ada
        categories = sorted(self.df['Product_Category'].unique())
        print(f"  Kategori: {', '.join(categories[:10])}")
        if len(categories) > 10:
            print(f"  ... dan {len(categories)-10} kategori lainnya")
        
        return self
    
    def standardize_data_types(self):
        """
        Tahap 6: Standarisasi tipe data
        """
        self.df['CustomerID'] = self.df['CustomerID'].astype(int)
        self.df['Transaction_ID'] = self.df['Transaction_ID'].astype(int)
        self.df['Product_Category'] = self.df['Product_Category'].astype(str)
        self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce')
        self.df['Avg_Price'] = pd.to_numeric(self.df['Avg_Price'], errors='coerce')
        self.df['TotalPrice'] = pd.to_numeric(self.df['TotalPrice'], errors='coerce')
        
        # Hapus baris dengan tipe data yang bermasalah setelah konversi
        self.df = self.df.dropna(subset=['Quantity', 'Avg_Price', 'TotalPrice'])
        
        print(f"✓ Tipe Data Distandarisasi:")
        print(f"  - CustomerID: int")
        print(f"  - Transaction_ID: int")
        print(f"  - Transaction_Date: datetime64")
        print(f"  - Product_Category: str")
        print(f"  - Quantity: float")
        print(f"  - Avg_Price: float")
        print(f"  - TotalPrice: float")
        
        return self
    
    def set_reference_date(self, date=None):
        """
        Set tanggal referensi untuk perhitungan Recency
        Jika tidak ditentukan, gunakan tanggal transaksi terakhir + 1 hari
        """
        if date is None:
            self.reference_date = self.df['Transaction_Date'].max() + pd.Timedelta(days=1)
        else:
            self.reference_date = pd.to_datetime(date)
            
        print(f"✓ Tanggal Referensi: {self.reference_date.strftime('%Y-%m-%d')}")
        
        return self
    
    def get_processed_data(self):
        """
        Mengembalikan data yang telah diproses
        """
        return self.df
    
    def get_summary(self):
        """
        Mengembalikan ringkasan hasil preprocessing
        """
        return {
            'original_rows': self.original_shape[0],
            'original_cols': self.original_shape[1],
            'final_rows': len(self.df),
            'final_cols': len(self.df.columns),
            'customers': self.df['CustomerID'].nunique(),
            'transactions': self.df['Transaction_ID'].nunique(),
            'categories': self.df['Product_Category'].nunique(),
            'date_range': f"{self.df['Transaction_Date'].min().strftime('%Y-%m-%d')} to {self.df['Transaction_Date'].max().strftime('%Y-%m-%d')}",
            'total_revenue': self.df['TotalPrice'].sum(),
            'reference_date': self.reference_date
        }
    
    def run_all(self):
        """
        Menjalankan seluruh tahap pra-pemrosesan sesuai metodologi (Subbab 3.8.1)
        """
        print("\n" + "="*70)
        print("PRA-PEMROSESAN DATA (Subbab 3.8.1)")
        print("="*70 + "\n")
        
        (self
         .clean_data()
         .handle_missing_values()
         .select_features()
         .create_derived_variables()
         .standardize_categories()
         .standardize_data_types()
         .set_reference_date())
        
        print("\n" + "="*70)
        print("RINGKASAN HASIL PRA-PEMROSESAN")
        print("="*70)
        summary = self.get_summary()
        print(f"Data Awal    : {summary['original_rows']:,} baris × {summary['original_cols']} kolom")
        print(f"Data Bersih  : {summary['final_rows']:,} baris × {summary['final_cols']} kolom")
        print(f"Data Retained: {summary['final_rows']/summary['original_rows']*100:.1f}%")
        print(f"\nPelanggan Unik  : {summary['customers']:,}")
        print(f"Transaksi Unik  : {summary['transactions']:,}")
        print(f"Kategori Produk : {summary['categories']}")
        print(f"Periode Data    : {summary['date_range']}")
        print(f"Total Revenue   : ${summary['total_revenue']:,.2f}")
        print("="*70 + "\n")
        
        return self