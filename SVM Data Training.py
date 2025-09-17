import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load dataset EEG
df = pd.read_csv('../FINAL DATASETS/frontal_raw_data.csv')
print(df.shape)

# 2. Pisahkan fitur dan label
X = df.drop('label', axis=1).values
y = df['label'].values

print(X.shape)

# 3. Bagi dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# 4. Buat pipeline: normalisasi + SVM
pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=1.0, gamma='scale')
)

# 5. Latih model
pipeline.fit(X_train, y_train)

# 6. Evaluasi model
y_pred = pipeline.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# 7. Simpan model
joblib.dump(pipeline, '../FINAL DATASETS/frontal_svm_eeg_model.pkl')
print("Model saved as 'frontal_svm_eeg_model.pkl'")
