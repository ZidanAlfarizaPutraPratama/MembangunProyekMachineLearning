import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat dataset hasil clustering
data = pd.read_csv('../notebooks/clustered_data.csv') 

# Memeriksa kolom yang tersedia
print("Kolom dalam dataset:", data.columns)

# Set style untuk seaborn
sns.set(style='whitegrid')

# Memastikan kolom 'Cluster' ada sebelum melanjutkan
if 'Cluster' not in data.columns:
    print("Kolom 'Cluster' tidak ditemukan dalam dataset. Pastikan proses clustering telah dilakukan dan kolom ini disimpan.")
else:
    # Visualisasi distribusi umur
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Age'], bins=20, kde=True)
    plt.title('Distribusi Umur Pelanggan')
    plt.xlabel('Usia')
    plt.ylabel('Frekuensi')
    plt.show()

    # Visualisasi pendapatan tahunan
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Annual Income ($)'], bins=20, kde=True)
    plt.title('Distribusi Pendapatan Tahunan Pelanggan')
    plt.xlabel('Pendapatan Tahunan ($)')
    plt.ylabel('Frekuensi')
    plt.show()

    # Visualisasi skor pengeluaran
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Spending Score (1-100)'], bins=20, kde=True)
    plt.title('Distribusi Skor Pengeluaran Pelanggan')
    plt.xlabel('Skor Pengeluaran (1-100)')
    plt.ylabel('Frekuensi')
    plt.show()

    # Visualisasi clustering
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Annual Income ($)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='Set2', s=100)
    plt.title('Segmentasi Pelanggan Berdasarkan Pendapatan dan Skor Pengeluaran')
    plt.xlabel('Pendapatan Tahunan ($)')
    plt.ylabel('Skor Pengeluaran (1-100)')
    plt.legend(title='Cluster')
    plt.show()
