import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat dataset hasil clustering
data = pd.read_csv('../data/hasil_clustering.csv')

# Memeriksa kolom yang tersedia
print("Kolom dalam dataset:", data.columns)

# Set style untuk seaborn
sns.set(style='whitegrid')

# Memastikan kolom 'Cluster' ada sebelum melanjutkan
if 'Cluster' not in data.columns:
    print("Kolom 'Cluster' tidak ditemukan dalam dataset.")
else:
    # Visualisasi distribusi usia
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Age', bins=20, kde=True, hue='Cluster', multiple='stack', legend=True)
    plt.title('Distribusi Usia Berdasarkan Cluster')
    plt.xlabel('Usia')
    plt.ylabel('Frekuensi')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Visualisasi distribusi Stress Level
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Stress_Level', bins=20, kde=True, hue='Cluster', multiple='stack', legend=True)
    plt.title('Distribusi Tingkat Stres Berdasarkan Cluster')
    plt.xlabel('Tingkat Stres')
    plt.ylabel('Frekuensi')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Visualisasi distribusi Work Life Balance Rating
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Work_Life_Balance_Rating', bins=20, kde=True, hue='Cluster', multiple='stack', legend=True)
    plt.title('Distribusi Rating Keseimbangan Kerja dan Kehidupan Berdasarkan Cluster')
    plt.xlabel('Rating Keseimbangan Kerja/Kehidupan')
    plt.ylabel('Frekuensi')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Visualisasi distribusi Social Isolation Rating
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Social_Isolation_Rating', bins=20, kde=True, hue='Cluster', multiple='stack', legend=True)
    plt.title('Distribusi Rating Isolasi Sosial Berdasarkan Cluster')
    plt.xlabel('Rating Isolasi Sosial')
    plt.ylabel('Frekuensi')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Visualisasi clustering dalam scatter plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Age', y='Stress_Level', hue='Cluster', data=data, palette='Set2', s=100)
    plt.title('Segmentasi Berdasarkan Usia dan Tingkat Stres')
    plt.xlabel('Usia')
    plt.ylabel('Tingkat Stres')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
