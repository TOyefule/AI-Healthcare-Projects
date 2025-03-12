import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load claims data
claims_data = pd.read_csv('claims_data.csv')

# Preprocessing claims data (e.g., removing missing values)
claims_data.fillna(claims_data.mean(), inplace=True)

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(claims_data.drop('fraud_label', axis=1))

# Train-test split
X = scaled_data
y = claims_data['fraud_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Isolation Forest model for anomaly detection
model = IsolationForest(contamination=0.05)  # Assuming 5% fraud rate
model.fit(X_train)

# Predict on test data
y_pred = model.predict(X_test)

# Convert predictions (-1: outlier, 1: inlier) to fraud labels
y_pred = np.where(y_pred == -1, 1, 0)

# Evaluate model performance
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))