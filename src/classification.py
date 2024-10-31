import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Membaca dataset yang sudah dilabeli
file_path = '../data/clustered_data.csv'
data = pd.read_csv(file_path)

# Mendefinisikan kolom numerik
numerical_columns = ['2020', '2021', '2022', '2023']

# Memilih fitur dan label
features = data[numerical_columns]  # Hanya kolom numerik untuk fitur
labels = data['Cluster']  # Label dari hasil clustering

# Memisahkan dataset menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standarisasi data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Menggunakan Random Forest untuk klasifikasi
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_scaled, y_train)

# Memprediksi label pada set testing
y_pred = classifier.predict(X_test_scaled)

# Menghitung akurasi dan F1-Score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Menampilkan hasil
print(f"Akurasi: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")

# Memastikan nilai akurasi dan F1-Score memenuhi syarat
if accuracy >= 0.87 and f1 >= 0.87:
    print("Akurasi dan F1-Score memenuhi syarat!")
else:
    print("Akurasi atau F1-Score belum memenuhi syarat.")
