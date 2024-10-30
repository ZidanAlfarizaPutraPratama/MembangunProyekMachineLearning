import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Memuat dataset
data = pd.read_csv('../data/customers.csv')

# Tampilkan kolom yang ada
print(data.columns)

# Fitur dan label
X = data[['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Gender']]
y = data['Gender']

# One-hot encoding untuk variabel kategorikal
X = pd.get_dummies(X, drop_first=True)

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Random Forest
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_weighted', cv=5)
grid_search.fit(X_train, y_train)

# Dapatkan model terbaik
best_model = grid_search.best_estimator_

# Prediksi
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Hitung akurasi dan F1-Score
train_accuracy = accuracy_score(y_train, y_pred_train)
train_f1 = f1_score(y_train, y_pred_train, average='weighted')
test_accuracy = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test, average='weighted')

# hasil
print(f'Train Accuracy: {train_accuracy}')
print(f'Train F1-Score: {train_f1}')
print(f'Test Accuracy: {test_accuracy}')
print(f'Test F1-Score: {test_f1}')

# laporan klasifikasi
print("\nLaporan Klasifikasi (Test Set):")
print(classification_report(y_test, y_pred_test))

# Console
if train_accuracy >= 0.87 and train_f1 >= 0.87 and test_accuracy >= 0.87 and test_f1 >= 0.87:
    print("Kriteria 6 terpenuhi dengan sempurna.")
else:
    print("Kriteria 6 belum terpenuhi.")
