
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


---
**break dulu**
## 🧩 Contoh / Rumus / Kode
### Sel 1: Inisialisasi Environment & SparkSession
	Langkah pertama adalah menyiapkan sesi Spark. Jika kamu menggunakan Google Colab, pastikan sudah menginstal PySpark terlebih dahulu.
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
	Penjelasan: `SparkSession` adalah pintu masuk utama. `.master("local[*]")` memerintahkan Spark untuk menggunakan semua core CPU yang tersedia di mesin lokal kamu.


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
	Penjelasan: Data dimuat dalam format Parquet karena performanya yang tinggi dan skema kolom yang terjaga


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
	Penjelasan: Kita menghitung total belanja, jumlah transaksi, rata-rata kuantitas, dan variasi produk. Nilai `null` pada usia diisi dengan 0 agar algoritma tidak error.


### **Sel 4: Vector Assembler & Standard Scaling**
	K-Means sangat sensitif terhadap skala data, jadi standarisasi wajib dilakukan.
```
from pyspark.ml.feature import VectorAssembler, StandardScaler

# Daftar kolom fitur numerik
input_cols = ["total_spent", "num_tx", "avg_qty", "unique_products", "age"]

# Gabungkan kolom menjadi satu vektor
assembler = VectorAssembler(inputCols=input_cols, outputCol="features_raw")
assembled = assembler.transform(features_df)

# Standarisasi fitur (Mean = 0, Std = 1)
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
scaler_model = scaler.fit(assembled)
assembled_scaled = scaler_model.transform(assembled)

assembled_scaled.select("customer_id", "features").show(5, truncate=False)
```
	Penjelasan: `VectorAssembler` menggabungkan kolom menjadi satu kolom vektor. `StandardScaler` penting karena jika satu fitur (misal: total belanja) punya angka ribuan dan fitur lain (misal: jumlah transaksi) hanya satuan, K-Means akan bias ke angka yang besar.
	
### **Sel 5: Mencari K Terbaik (Elbow & Silhouette)**
	Kita akan mencoba beberapa nilai k (jumlah klaster) untuk melihat mana yang paling optimal.
```
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

data_for_kmeans = assembled_scaled.select("features")
evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")

results = []
for k in range(2, 7):
    km = KMeans(k=k, seed=42, featuresCol="features", predictionCol="prediction")
    model = km.fit(data_for_kmeans)
    
    # Hitung Silhouette Score
    predictions = model.transform(data_for_kmeans)
    silhouette = evaluator.evaluate(predictions)
    
    # Ambil WSSSE (Within-Cluster SSE)
    wssse = model.summary.trainingCost
    
    results.append((k, wssse, silhouette))
    print(f"k={k}: WSSSE={wssse:.2f}, Silhouette={silhouette:.4f}")
```
	Penjelasan: WSSSE digunakan untuk metode "Elbow" (cari tekukan grafik), sedangkan Silhouette mengukur seberapa baik sebuah titik masuk ke klasternya sendiri dibanding klaster lain (range -1 hingga 1, makin tinggi makin baik).


### **Sel 6: Training Model Final & Visualisasi PCA**
	Kita pilih k terbaik secara otomatis dan melakukan reduksi dimensi untuk visualisasi.
```
import matplotlib.pyplot as plt
from pyspark.ml.feature import PCA

# Pilih k dengan silhouette tertinggi
best_k = sorted(results, key=lambda x: x[2], reverse=True)[0][0]
print(f"\nBest k terpilih: {best_k}")

# Fit model final
best_model = KMeans(k=best_k, seed=42, featuresCol="features").fit(data_for_kmeans)
clustered = best_model.transform(assembled_scaled)

# PCA untuk visualisasi 2D
pca = PCA(k=2, inputCol="features", outputCol="pca2")
viz_df = pca.fit(assembled_scaled).transform(clustered)

# Ambil data ke driver untuk plotting (Gunakan limit jika data besar)
points = [(row.pca2[0], row.pca2[1], int(row.prediction)) for row in viz_df.select("pca2", "prediction").limit(1000).collect()]
xs, ys, cs = zip(*points)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(xs, ys, c=cs, cmap='viridis')
plt.colorbar(scatter, label='Cluster ID')
plt.title(f"K-Means Clusters (k={best_k}) PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```
	Penjelasan: Karena kita punya 5 fitur, kita tidak bisa menggambarnya di grafik biasa. PCA mereduksi 5 fitur tersebut menjadi 2 komponen utama (PC1 & PC2) agar bisa di-plot secara visual.


### **Sel 7: Mengembalikan Centroid ke Skala Asli**
	Untuk memahami profil bisnis (misal: "Klaster 0 belanja rata-rata berapa Rupiah?"), kita harus membalikkan angka standarisasi tadi.
```
means = scaler_model.mean.toArray()
stds = scaler_model.std.toArray()

print(f"Profil Centroid (Skala Asli) - Urutan: {input_cols}")
for idx, center in enumerate(best_model.clusterCenters()):
    original_center = [round(center[i] * stds[i] + means[i], 2) for i in range(len(center))]
    print(f"Cluster {idx}: {original_center}")
```
	Penjelasan: Rumus baliknya adalah x=x′⋅σ+μ. Ini mempermudah interpretasi hasil bagi orang bisnis.


### Sel 8: Simpan Model & Hasil
	Simpan dalam megatech/.data
```
OUT_DIR = "/content/megatech/.data/out_p12"

# Simpan model
best_model.write().overwrite().save(f"{OUT_DIR}/kmeans_model_k{best_k}")

# Simpan label pelanggan
(clustered.select("customer_id", "prediction")
 .write.mode("overwrite").parquet(f"{OUT_DIR}/customer_clusters.parquet"))

print("Selesai! Model dan label tersimpan di:", OUT_DIR)
```


## Penjelasan Singkat:
1. Scaling itu Wajib: K-Means menggunakan jarak Euclidean. Jika satu fitur punya angka jutaan (rupiah) dan yang lain puluhan (usia), fitur rupiah akan mendominasi perhitungan jarak.
2. Evaluasi: Kamu menggunakan dua metrik. WSSSE untuk melihat kepadatan internal klaster, dan Silhouette untuk melihat pemisahan antar klaster.
3. PCA: Ini hanya alat bantu visual. PCA membantu kita melihat apakah klaster-klaster tersebut benar-benar terpisah secara visual dalam ruang 2 dimensi.
4. Inverse Scaling: Centroid yang dihasilkan K-Means awalnya dalam bentuk angka standar (seperti -0.5, 1.2). Kita perlu mengembalikannya ke satuan asli (Rupiah, tahun, jumlah) agar bisa dibaca maknanya.
## 🛠️ Checklist Akhir
- [ ] SparkSession berhasil dibuat.
- [ ] Fitur sudah digabung menjadi VectorUDT.
- [ ] StandardScaler sudah diterapkan dengan withMean=True.
- [ ] Nilai k terbaik dipilih berdasarkan Silhouette tertinggi.
- [ ] Model dan label klaster sudah disimpan ke format Parquet.
## 🔗 Referensi
- jangan lupa referensi nya
- (link drive modul)

