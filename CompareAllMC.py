import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Load data
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

X_train = train_df.drop(columns=["label"]).values.astype(np.float32) / 255.0
y_train = train_df["label"].values
X_test = test_df.drop(columns=["label"]).values.astype(np.float32) / 255.0
y_test = test_df["label"].values

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, solver='adam', random_state=42)
}

# Train and collect misclassified examples
misclassified_examples = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    wrong = np.where(y_pred != y_test)[0]
    misclassified_examples[name] = (y_test[wrong], y_pred[wrong], X_test[wrong])

# Plot misclassified examples side-by-side
for name in models.keys():
    true_labels, pred_labels, images = misclassified_examples[name]
    plt.figure(figsize=(10, 2))
    for i in range(5):
        img = images[i].reshape(28, 28)
        true = true_labels[i]
        pred = pred_labels[i]
        plt.subplot(1, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"T:{true} P:{pred}")
        plt.axis('off')
    plt.suptitle(f"Misclassified Examples – {name}")
    plt.tight_layout()
    plt.show()
