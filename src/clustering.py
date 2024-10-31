import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Membaca dataset
file_path = '../data/combined_financial_data_idx.csv' 
data = pd.read_csv(file_path)

# Tampilkan kolom dalam dataset
print("Kolom dalam dataset:")
print(data.columns)

# Memilih kolom kategorikal dan numerikal
categorical_column = 'type' 
numerical_columns = ['2020', '2021', '2022', '2023']  

# Memastikan kolom yang dipilih ada dalam dataset
if categorical_column not in data.columns or not all(col in data.columns for col in numerical_columns):
    raise KeyError(f"Kolom '{categorical_column}' atau salah satu kolom numerik tidak ditemukan dalam dataset.")

# Menyiapkan data untuk clustering
X = data[[categorical_column] + numerical_columns].copy()

# Mengubah kolom kategorikal menjadi numerik
X[categorical_column] = X[categorical_column].astype('category').cat.codes

# Mengisi nilai NaN dengan rata-rata kolom
X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].mean())

# Standarisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Memulai proses clustering
print("Memulai proses clustering...")
kmeans = KMeans(n_clusters=3, random_state=42)  
labels = kmeans.fit_predict(X_scaled)

# Menghitung Silhouette Score
silhouette_avg = silhouette_score(X_scaled, labels)

# Mengurangi dimensi untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Menambahkan label cluster ke DataFrame
data['Cluster'] = labels

# Tampilkan data dengan label cluster
print("Data dengan Label Cluster:")
print(data.head())

# Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Hasil Clustering dengan KMeans')
plt.xlabel('Komponen PCA 1')
plt.ylabel('Komponen PCA 2')
plt.colorbar(label='Label Cluster')
plt.show()

# Menyimpan hasil ke CSV    
output_file = '../data/clustered_data.csv'  
data.to_csv(output_file, index=False)

# Menampilkan Silhouette Score di akhir
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Hasil telah disimpan ke {output_file}.")
