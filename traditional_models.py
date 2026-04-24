import numpy as np
import os

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

BASE_PATH = "datasets/Earthquakes"
OUTPUT_DIR = "train/traditional_models"


def save_result(filename, content):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w") as f:
        f.write(content)

    print(f"Saved: {path}")


def load_data():
    X_train = np.load(os.path.join(BASE_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(BASE_PATH, "y_train.npy"))
    X_test = np.load(os.path.join(BASE_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(BASE_PATH, "y_test.npy"))

    return X_train, y_train, X_test, y_test


def run_dtw(X_train, y_train, X_test, y_test):
    model = KNeighborsTimeSeriesClassifier(n_neighbors=7, metric="dtw")
    model.fit(X_train, y_train, )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    content = f"""DTW (k=1)
Train Accuracy: {train_acc:.4f}
Test Accuracy : {test_acc:.4f}
"""

    save_result("dtw.txt", content)


def run_euclidean_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="euclidean")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    content = f"""Euclidean KNN (k=3)
Train Accuracy: {train_acc:.4f}
Test Accuracy : {test_acc:.4f}
"""
    save_result("euclidean_knn.txt", content)


def run_random_forest(X_train, y_train, X_test, y_test):
    X_train_flat = X_train.squeeze()
    X_test_flat = X_test.squeeze()

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_flat, y_train)

    y_train_pred = model.predict(X_train_flat)
    y_test_pred = model.predict(X_test_flat)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    content = f"""Random Forest
Train Accuracy: {train_acc:.4f}
Test Accuracy : {test_acc:.4f}
"""
    save_result("random_forest.txt", content)


def run_svm(X_train, y_train, X_test, y_test):
    X_train_flat = X_train.squeeze()
    X_test_flat = X_test.squeeze()

    model = SVC(kernel="rbf")
    model.fit(X_train_flat, y_train)

    y_train_pred = model.predict(X_train_flat)
    y_test_pred = model.predict(X_test_flat)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    content = f"""SVM (RBF)
Train Accuracy: {train_acc:.4f}
Test Accuracy : {test_acc:.4f}
"""
    save_result("svm.txt", content)


def main():
    print("Loading dataset...\n")

    X_train, y_train, X_test, y_test = load_data()

    print("Running traditional models...\n")

    run_dtw(X_train, y_train, X_test, y_test)
    run_euclidean_knn(X_train, y_train, X_test, y_test)
    run_random_forest(X_train, y_train, X_test, y_test)
    run_svm(X_train, y_train, X_test, y_test)

    print("\nAll results saved.")


if __name__ == "__main__":
    main()
