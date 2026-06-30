# Smart AI-Powered Fraud Detection System

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Step 2: Load Dataset
data = pd.read_csv('fraud_data.csv')  # Place dataset in working directory

# Step 3: Explore Dataset
print(data.head())
print(data.info())
print(data.describe())
print(data['Fraud'].value_counts())

# Step 4: Data Preprocessing
# Handle missing values
data = data.dropna()

# Encode categorical variables
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    if col != 'Fraud':
        data[col] = le.fit_transform(data[col])

# Encode target variable
data['Fraud'] = data['Fraud'].map({'No':0, 'Yes':1})

# Feature scaling
scaler = StandardScaler()
features = data.drop('Fraud', axis=1)
features_scaled = scaler.fit_transform(features)

# Step 5: Handle Class Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features_scaled, data['Fraud'])

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 7: Train Models
# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Step 8: Evaluate Models
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC-AUC Score
roc = roc_auc_score(y_test, y_pred_rf)
print("Random Forest ROC-AUC Score:", roc)