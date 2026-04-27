from tslearn.datasets import UCR_UEA_datasets
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DATASET_PATH = "datasets"
DATASET_NAME = "Earthquakes"


def main():
    print("Loading dataset...\n")

    datasets = UCR_UEA_datasets()

    os.makedirs(BASE_DATASET_PATH, exist_ok=True)

    dataset_path = os.path.join(BASE_DATASET_PATH, DATASET_NAME)
    os.makedirs(dataset_path, exist_ok=True)

    X_train_path = os.path.join(dataset_path, "X_train.npy")
    y_train_path = os.path.join(dataset_path, "y_train.npy")
    X_test_path = os.path.join(dataset_path, "X_test.npy")
    y_test_path = os.path.join(dataset_path, "y_test.npy")

    if all(os.path.exists(p) for p in [X_train_path, y_train_path, X_test_path, y_test_path]):
        print("Loading dataset from local files...\n")

        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

    else:
        print("Downloading dataset...\n")

        X_train, y_train, X_test, y_test = datasets.load_dataset(DATASET_NAME)

        np.save(X_train_path, X_train)
        np.save(y_train_path, y_train)
        np.save(X_test_path, X_test)
        np.save(y_test_path, y_test)

        print("Dataset saved locally\n")

    print(f"Dataset: {DATASET_NAME}")
    print("=" * 40)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    print(f"\nNumber of classes: {len(np.unique(y_train))}")
    print(f"Classes: {np.unique(y_train)}")

    print("\nClass distribution (Train)")
    print("=" * 40)

    unique, counts = np.unique(y_train, return_counts=True)

    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples")

    print(f"\nLength of each time series: {X_train.shape[1]}")

    sample_index = 0
    sample = X_train[sample_index].ravel()
    label = y_train[sample_index]

    print("\nSample Example")
    print("=" * 40)
    print(f"Label: {label}")
    print(f"First 10 values: {sample[:10]}")
    print(f"Min value: {np.min(sample)}")
    print(f"Max value: {np.max(sample)}")

    plot_path = os.path.join(dataset_path, "sample_plot.png")

    plt.plot(sample)
    plt.title(f"Sample Time Series (Label: {label})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.savefig(plot_path)

    print(f"Saved plot at {plot_path}")


if __name__ == "__main__":
    main()
