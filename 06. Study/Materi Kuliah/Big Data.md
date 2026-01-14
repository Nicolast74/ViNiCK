
# 📚 Study: 
Praktikum week 12

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
- kalau ada rumus atau code kasih sini aja


## 🛠️ Checklist Akhir
- [ ] SparkSession berhasil dibuat.
- [ ] Fitur sudah digabung menjadi VectorUDT.
- [ ] StandardScaler sudah diterapkan dengan withMean=True.
- [ ] Nilai k terbaik dipilih berdasarkan Silhouette tertinggi.
- [ ] Model dan label klaster sudah disimpan ke format Parquet.
## 🔗 Referensi
- jangan lupa referensi nya

