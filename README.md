# Proyek Analisis Data Pelanggan untuk Membangun Proyek Machine Learning

Proyek ini bertujuan untuk menganalisis data pelanggan dan mengklasifikasikan mereka menggunakan algoritma Random Forest dan metode clustering.

## Daftar Isi
- [Pendahuluan](#pendahuluan)
- [Struktur Proyek](#struktur-proyek)
- [Fitur](#fitur)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Evaluasi Model](#evaluasi-model)
- [Kontribusi](#kontribusi)
- [Lisensi](#lisensi)

## Pendahuluan

Proyek ini menggunakan data pelanggan untuk memahami perilaku konsumen dan melakukan segmentasi. Dengan menggunakan algoritma klasifikasi dan clustering, kita dapat mengidentifikasi jenis kelamin pelanggan berdasarkan fitur lain seperti usia, pendapatan tahunan, dan skor pengeluaran.

## Struktur Proyek

Berikut adalah struktur direktori dari proyek ini:

```
.
├── data
│   └── Customers.csv
├── notebooks
│   ├── [Clustering]_Submission_Akhir_BMLP_Zidannnapp.ipynb
│   ├── [Klasifikasi]_Submission_Akhir_BMLP_ZIdannnapp.ipynb
│   ├── clustered_data.csv
│   └── customers_with_clusters.csv
├── results
│   └── clustering_results.csv
├── src
│   ├── classification.py
│   └── clustering.py
└── visualizations
    └── visualization.py
```

## Fitur

- Analisis data pelanggan berdasarkan usia, pendapatan tahunan, dan skor pengeluaran.
- Penggunaan algoritma klasifikasi seperti Random Forest untuk mengklasifikasikan jenis kelamin pelanggan.
- Metode clustering untuk mengelompokkan pelanggan.
- Visualisasi hasil analisis dan clustering.

## Instalasi

Langkah-langkah untuk menginstal dependensi yang diperlukan:

1. Kloning repository ini:
   ```bash
   git clone <URL_REPOSITORY>
   ```
2. Masuk ke direktori proyek:
   ```bash
   cd project-folder
   ```
3. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Penggunaan

Berikut adalah langkah-langkah untuk menjalankan file dalam proyek ini:

### 1. Menjalankan Clustering

Untuk menjalankan analisis clustering, gunakan notebook berikut:

```bash
cd notebooks
jupyter notebook [Clustering]_Submission_Akhir_BMLP_Zidannnapp.ipynb
```

Notebook ini akan menghasilkan file `clustered_data.csv` dan `customers_with_clusters.csv` di direktori `notebooks`.

### 2. Menjalankan Klasifikasi

Setelah melakukan clustering, Anda dapat melakukan klasifikasi dengan menggunakan notebook berikut:

```bash
jupyter notebook [Klasifikasi]_Submission_Akhir_BMLP_ZIdannnapp.ipynb
```

Notebook ini akan menggunakan hasil clustering untuk melatih model klasifikasi dan menyimpan hasilnya.

### 3. Menjalankan Skrip Python

Jika Anda ingin menjalankan analisis menggunakan skrip Python, Anda bisa menjalankan:

- Untuk klasifikasi:
   ```bash
   python src/classification.py
   ```

- Untuk clustering:
   ```bash
   python src/clustering.py
   ```

- Untuk visualisasi hasil:
   ```bash
   python visualizations/visualization.py
   ```

## Evaluasi Model

Hasil evaluasi model yang digunakan dalam proyek ini:

- **Akurasi**: 92%
- **F1-Score**: 91%

## Kontribusi

Jika Anda ingin berkontribusi, silakan ikuti langkah-langkah berikut:

1. Fork repository ini.
2. Buat branch baru:
   ```bash
   git checkout -b feature-branch
   ```
3. Lakukan perubahan dan commit:
   ```bash
   git commit -m "Deskripsi perubahan"
   ```
4. Push ke branch Anda:
   ```bash
   git push origin feature-branch
   ```
5. Buat Pull Request.

## Lisensi

Proyek ini dilisensikan di bawah [Nama Lisensi] - lihat file [LICENSE](LICENSE) untuk rincian lebih lanjut.
