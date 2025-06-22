import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the MNIST data
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# Prepare features and labels
X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
y_train = train_df["label"].values

X_test = test_df.drop(columns=["label"]).values.astype(np.float32)
y_test = test_df["label"].values

# Normalize pixel values to range [0, 1]
X_train /= 255.0
X_test /= 255.0

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Sum importances across all trees
importances = model.feature_importances_.reshape(28, 28)

plt.figure(figsize=(6, 6))
plt.imshow(importances, cmap='hot', interpolation='nearest')
plt.title("Pixel Feature Importances (Random Forest)")
plt.axis('off')
plt.colorbar()
plt.show()