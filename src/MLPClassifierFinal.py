import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the MNIST data
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# Prepare features and labels
X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
y_train = train_df["label"].values

X_test = test_df.drop(columns=["label"]).values.astype(np.float32)
y_test = test_df["label"].values

# Normalize pixel values to [0, 1]
X_train /= 255.0
X_test /= 255.0

# Initialize and train MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, solver='adam', random_state=42, verbose=False)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Visualizing Training Loss Curve
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
