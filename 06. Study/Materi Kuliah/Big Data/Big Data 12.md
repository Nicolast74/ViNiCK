
# 📚 Study: 
Praktikum & teori week 12

## 🎯 Apa yang Dipelajari
- PySpark K-Means Clustering

## 📝 Catatan Inti
🏗️ Alur Kerja Utama

1. Setup Lingkungan: Inisialisasi SparkSession sebagai pintu masuk API MLlib.

2. Feature Engineering: Membangun profil pelanggan dari data transaksi (agregasi total_spent, num_tx, dll.).

3. Preprocessing: Menggabungkan kolom menjadi vektor (VectorAssembler) dan melakukan standardisasi (StandardScaler).

4. Model Training: Menjalankan algoritma K-Means dengan berbagai nilai k.

5. Evaluasi: Menggunakan metode Elbow (WSSSE) dan Silhouette Score untuk memilih jumlah klaster terbaik.

6. Visualisasi: Mereduksi dimensi dengan PCA untuk melihat sebaran klaster dalam 2D.

💡 Konsep Penting
1. Mengapa Harus Scaling?
	- K-Means sangat sensitif terhadap skala data.
	- Fitur dengan rentang nilai besar (misal: Total Belanja) akan mendominasi perhitungan jarak dibanding fitur kecil (misal: Usia) jika tidak distandarisasi.
	- Rumus Standardisasi: x′=(x−μ)/σ
2. Metrik Evaluasi

|Metrik|Nama Lengkap|Interpretasi|
|---|---|---|
|**WSSSE**|_Within-cluster Sum of Squared Errors_|Mengukur kepadatan internal klaster. Semakin kecil nilainya, semakin baik.|

3. PCA (Principal Component Analysis)
- Digunakan untuk reduksi dimensi fitur menjadi 2 komponen utama (PC1 & PC2).
- Tujuannya agar data multidimensi dapat divisualisasikan dalam plot 2D.

⚠️ Pitfalls & Troubleshooting
- Penanganan Null: 
  Nilai hilang pada fitur numerik harus diisi 
  (misal: na.fill({"age":0})) agar algoritma tidak gagal.
- Collect Data: 
  Hindari penggunaan .collect() pada DataFrame yang sangat besar karena dapat menyebabkan crash pada driver; 
  gunakan .sample() atau .limit() untuk plotting.
- Interpretasi Bisnis: 
  Centroid awal berada dalam skala standar. 
  Gunakan rumus balik x=x′⋅σ+μ untuk mengembalikan nilai ke satuan asli (Rupiah, tahun, dsb.).
## 🧩 Contoh / Rumus / Kode
### Sel 1: Inisialisasi Environment & SparkSession
	Langkah pertama adalah menyiapkan sesi Spark. Jika kamu menggunakan Google Colab, pastikan sudah menginstal PySpark terlebih dahulu.
Python
```
# Install pyspark jika belum ada
# %pip install pyspark==3.5.1

from pyspark.sql import SparkSession

# Inisialisasi SparkSession
spark = (SparkSession.builder
         .appName("MegaTech_P12_KMeans_Clustering")
         .master("local[*]")
         .getOrCreate())

print("Spark Version:", spark.version)

    Penjelasan: SparkSession adalah pintu masuk utama. .master("local[*]") memerintahkan Spark untuk menggunakan semua core CPU yang tersedia di mesin lokal kamu.
```

### **Sel 2: Load Data dari Warehouse**
	Kita akan memuat data transaksi (`sales_enriched`) dan data pelanggan (`customers`).
```
WH_DIR = "/content/megatech/.data/warehouse"

# Memuat data Parquet
try:
    sales = spark.read.parquet(f"{WH_DIR}/sales_enriched.parquet")
    cust = spark.read.parquet(f"{WH_DIR}/customers.parquet")
    
    sales.printSchema()
    cust.printSchema()
except Exception as e:
    print("Error: Pastikan path warehouse benar!", e)
```

### **Sel 3: Agregasi Fitur Pelanggan**
	Kita akan mengubah data transaksi mentah menjadi profil perilaku pelanggan (RFM-like features).

```
from pyspark.sql.functions import sum as _sum, count as _count, avg as _avg, countDistinct

# Agregasi data transaksi per pelanggan
agg = (sales.groupBy("customer_id")
       .agg(_sum(sales.quantity * sales.price).alias("total_spent"),
            _count("*").alias("num_tx"),
            _avg("quantity").alias("avg_qty"),
            countDistinct("product").alias("unique_products")))

# Join dengan data demografi dan isi nilai null pada usia
features_df = (agg.join(cust.select("customer_id", "city", "age"), 
                        on="customer_id", how="left")
               .fill({"age": 0}))

features_df.show(5, truncate=False)
```

## 🛠️ Checklist Akhir
- [ ] SparkSession berhasil dibuat.
- [ ] Fitur sudah digabung menjadi VectorUDT.
- [ ] StandardScaler sudah diterapkan dengan withMean=True.
- [ ] Nilai k terbaik dipilih berdasarkan Silhouette tertinggi.
- [ ] Model dan label klaster sudah disimpan ke format Parquet.
## 🔗 Referensi
- jangan lupa referensi nya

