import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat dataset hasil clustering
data = pd.read_csv('../data/clustered_data.csv') 

# Memeriksa kolom yang tersedia
print("Kolom dalam dataset:", data.columns)

# Set style untuk seaborn
sns.set(style='whitegrid')

# Memastikan kolom 'Cluster' ada sebelum melanjutkan
if 'Cluster' not in data.columns:
    print("Kolom 'Cluster' tidak ditemukan dalam dataset.")
else:
    # Visualisasi distribusi data untuk tahun 2020
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='2020', bins=20, kde=True, hue='Cluster', multiple='stack', legend=True)
    plt.title('Distribusi Data Tahun 2020 Berdasarkan Cluster')
    plt.xlabel('Data 2020')
    plt.ylabel('Frekuensi')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Visualisasi distribusi data untuk tahun 2021
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='2021', bins=20, kde=True, hue='Cluster', multiple='stack', legend=True)
    plt.title('Distribusi Data Tahun 2021 Berdasarkan Cluster')
    plt.xlabel('Data 2021')
    plt.ylabel('Frekuensi')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Visualisasi distribusi data untuk tahun 2022
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='2022', bins=20, kde=True, hue='Cluster', multiple='stack', legend=True)
    plt.title('Distribusi Data Tahun 2022 Berdasarkan Cluster')
    plt.xlabel('Data 2022')
    plt.ylabel('Frekuensi')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Visualisasi distribusi data untuk tahun 2023
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='2023', bins=20, kde=True, hue='Cluster', multiple='stack', legend=True)
    plt.title('Distribusi Data Tahun 2023 Berdasarkan Cluster')
    plt.xlabel('Data 2023')
    plt.ylabel('Frekuensi')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Visualisasi clustering dalam scatter plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='2020', y='2021', hue='Cluster', data=data, palette='Set2', s=100)
    plt.title('Segmentasi Berdasarkan Data Tahun 2020 dan 2021')
    plt.xlabel('Data 2020')
    plt.ylabel('Data 2021')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
