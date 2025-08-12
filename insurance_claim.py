# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("insurance3r2.csv")

# Basic data inspection
print(data.head())
print(data.info())
print(data.describe())

# Drop missing values
data = data.dropna()

# Visualize class distribution
plt.title('Class Distributions \n (0: No Claim || 1: Claim)', fontsize=14)
sns.set(style="darkgrid")
sns.countplot(x='insuranceclaim', data=data)
plt.grid()
plt.show()

# Correlation matrix
corr = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Drop 'region' column (non-numeric and not encoded)
data = data.drop('region', axis=1)

# Bar plots for feature relationships
plt.figure(figsize=(16, 8))
sns.barplot(x='age', y='charges', data=data)
plt.title("Age vs Charges")
plt.show()

plt.figure(figsize=(6, 6))
sns.barplot(x='sex', y='charges', data=data)
plt.title('Sex vs Charges')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='children', y='charges', data=data)
plt.title('Children vs Charges')
plt.show()

plt.figure(figsize=(6, 6))
sns.barplot(x='smoker', y='charges', data=data)
plt.title('Smoker vs Charges')
plt.show()

# Encode categorical variables
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

# Split features and target
X = data.drop('insuranceclaim', axis=1)
Y = data['insuranceclaim']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Save processed data for reference
data.to_csv('finaldata.csv', index=False)
X_test.to_csv('testing.csv', index=False)

# Optional: Feature scaling (commented out unless needed)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Train Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions and evaluation
ypred = rf.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, ypred))
print("Classification Report:\n", classification_report(y_test, ypred))
print("Accuracy Score:", accuracy_score(y_test, ypred))

# Cross-validation
from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=rf, X=X_train, y=y_train, cv=10)
print("Cross-Validation Mean Accuracy:", acc.mean())
print("Cross-Validation Std Dev:", acc.std())

# ✅ Save model to disk
import pickle
pickle.dump(rf, open('model.pkl', 'wb'))

# Optional: Load model to verify
model = pickle.load(open('model.pkl', 'rb'))
print("Model loaded successfully.")
