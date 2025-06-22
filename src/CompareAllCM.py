import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Load and normalize data
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

X_train = train_df.drop(columns=["label"]).values.astype(np.float32) / 255.0
y_train = train_df["label"].values
X_test = test_df.drop(columns=["label"]).values.astype(np.float32) / 255.0
y_test = test_df["label"].values

# Define and train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, solver='adam', random_state=42)
}

# Create confusion matrices
plt.figure(figsize=(18, 5))

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.subplot(1, 3, i + 1)
    sns.heatmap(cm, annot=False, fmt='d', cmap='coolwarm', xticklabels=range(10), yticklabels=range(10))
    plt.title(f"{name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.suptitle("Confusion Matrices of Logistic Regression, Random Forest, and MLPClassifier")
plt.tight_layout()
plt.show()
