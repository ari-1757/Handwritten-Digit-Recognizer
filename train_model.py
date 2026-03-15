import numpy as np
import struct
import joblib

from sklearn import svm
from sklearn.metrics import accuracy_score


# -------- Load MNIST Images --------
def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8)
        images = images.reshape(num, rows * cols)
    return images


# -------- Load MNIST Labels --------
def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


print("Loading dataset...")

X_train = load_images("train-images.idx3-ubyte")
y_train = load_labels("train-labels.idx1-ubyte")

X_test = load_images("t10k-images.idx3-ubyte")
y_test = load_labels("t10k-labels.idx1-ubyte")

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)


# -------- Normalize Data --------
X_train = X_train / 255.0
X_test = X_test / 255.0


print("Training SVM model...")

model = svm.SVC(kernel='rbf')

model.fit(X_train, y_train)


print("Making predictions...")

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)


# -------- Save Model --------
joblib.dump(model, "digit_svm_model.pkl")

print("Model saved as digit_svm_model.pkl")