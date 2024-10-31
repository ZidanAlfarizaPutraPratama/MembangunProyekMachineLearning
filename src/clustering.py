import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer

# Langkah 1: Memuat Dataset
data = pd.read_csv('../data/Impact_of_Remote_Work_on_Mental_Health.csv')

# Memastikan dataset memiliki minimal 2500 baris
if data.shape[0] < 2500:
    raise ValueError("Dataset harus memiliki minimal 2500 baris.")

# Menampilkan kolom dalam dataset
print("Kolom dalam dataset:")
print(data.columns)

# Langkah 2: Memilih Fitur
# Memilih minimal 5 fitur, dengan 1 kolom kategorikal dan 1 kolom numerikal
features = ['Age', 'Stress_Level', 'Hours_Worked_Per_Week', 
            'Work_Life_Balance_Rating', 'Social_Isolation_Rating', 
            'Job_Role']

# Mengatasi Missing Values
data['Stress_Level'] = data['Stress_Level'].fillna(data['Stress_Level'].mode()[0])  # Mengisi dengan modus
data['Job_Role'] = data['Job_Role'].fillna(data['Job_Role'].mode()[0])  # Mengisi dengan modus

# Langkah 3: Memisahkan Fitur
X = data[features]

# Langkah 4: One-Hot Encoding untuk Kolom Kategorikal dan Normalisasi Data
numeric_features = ['Age', 'Hours_Worked_Per_Week', 'Social_Isolation_Rating']
categorical_features = ['Stress_Level', 'Work_Life_Balance_Rating', 'Job_Role']

# Membuat transformasi kolom
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Langkah 5: Clustering dengan KMeans
X_transformed = preprocessor.fit_transform(X)

silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_transformed)
    score = silhouette_score(X_transformed, kmeans.labels_)
    silhouette_scores.append(score)

best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f'Best number of clusters: {best_n_clusters}')

# Visualisasi Silhouette Score
plt.plot(range(2, 11), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Various Number of Clusters')
plt.grid()
plt.show()

# Menggunakan jumlah cluster terbaik untuk clustering
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_transformed)

# Langkah 6: Analisis Hasil Cluster
for cluster in range(best_n_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    print(f"\nAnalisis untuk Cluster {cluster}:")
    print(cluster_data.describe())

# Langkah 7: Klasifikasi
X_train, X_test, y_train, y_test = train_test_split(X_transformed, data['Cluster'], test_size=0.2, random_state=42)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)

# Evaluasi Klasifikasi
print('Random Forest Accuracy:', accuracy_score(y_test, rf_predictions))
print('Random Forest F1 Score:', f1_score(y_test, rf_predictions, average='weighted'))
print('Decision Tree Accuracy:', accuracy_score(y_test, dt_predictions))
print('Decision Tree F1 Score:', f1_score(y_test, dt_predictions, average='weighted'))
