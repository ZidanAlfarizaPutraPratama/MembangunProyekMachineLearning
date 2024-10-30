import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load your dataset
data = pd.read_csv('../data/customers.csv')  # Update path to your dataset

# Memilih fitur yang sesuai
categorical_columns = ['Gender']
numerical_columns = ['Annual Income ($)']

# Memilih fitur
X = data[categorical_columns + numerical_columns]

# Menangani nilai yang hilang
X = X.dropna()  # Menghapus baris dengan nilai yang hilang

# OneHotEncoding untuk fitur kategorikal
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_columns])

# Menggabungkan fitur yang telah di-encode dengan fitur numerik
X_numerical = X[numerical_columns].values
X_combined = np.hstack((X_encoded, X_numerical))

# Standardisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# PCA untuk reduksi dimensi (jika diperlukan)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# KMeans Clustering dan evaluasi silhouette score
best_score = -1
best_n_clusters = 0

for n_clusters in range(2, 15):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    silhouette_avg = silhouette_score(X_pca, labels)
    
    print(f'KMeans dengan {n_clusters} cluster -> Skor silhouette: {silhouette_avg:.4f}')
    
    # Memilih model dengan silhouette score terbaik
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_n_clusters = n_clusters

# Menampilkan hasil terbaik
print(f'Skor silhouette terbaik: {best_score:.4f} dengan {best_n_clusters} cluster.')

if best_score >= 0.55:
    print(f'Kualitas clustering memenuhi kriteria dengan silhouette score >= 0.55.')
else:
    print('Kualitas clustering tidak memenuhi kriteria.')

# Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.5, cmap='viridis')
plt.title('Hasil Clustering dengan KMeans')
plt.xlabel('Dimensi 1')
plt.ylabel('Dimensi 2')
plt.grid()
plt.show()
