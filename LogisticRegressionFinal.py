import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# Split into features and labels
X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
y_train = train_df["label"].values

X_test = test_df.drop(columns=["label"]).values.astype(np.float32)
y_test = test_df["label"].values

# Normalize pixel values to [0, 1]
X_train /= 255.0
X_test /= 255.0

# Create and train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate performance
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize misclassified digits
misclassified = np.where(y_test != y_pred)[0]

plt.figure(figsize=(10, 2))
for i, idx in enumerate(misclassified[:5]):
    img = X_test[idx].reshape(28, 28)
    true = y_test[idx]
    pred = y_pred[idx]
    plt.subplot(1, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"T:{true} P:{pred}")
    plt.axis('off')
plt.suptitle("Misclassified Test Samples")
plt.tight_layout()
plt.show()

