import numpy as np
import os
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField
from scipy.signal import spectrogram

BASE_PATH = "datasets/Earthquakes"
IMAGE_SIZE = 128


def save_image(img, path):
    plt.imshow(img, origin='lower')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_spectrogram(signal, path):
    f, t, Sxx = spectrogram(signal)
    plt.pcolormesh(t, f, Sxx)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_split(X, y, split_name, method_name):
    print(f"\nProcessing {method_name} - {split_name}")

    X = X.squeeze()

    base_dir = os.path.join(BASE_PATH, "images", method_name, split_name)

    for label in np.unique(y):
        os.makedirs(os.path.join(base_dir, str(label)), exist_ok=True)

    gaf = GramianAngularField(image_size=IMAGE_SIZE, method='summation')
    gadf = GramianAngularField(image_size=IMAGE_SIZE, method='difference')
    rp = RecurrencePlot()
    mtf = MarkovTransitionField(image_size=IMAGE_SIZE, n_bins=4)

    for i in range(len(X)):
        sample = X[i]
        label = y[i]

        if method_name == "gaf":
            img = gaf.fit_transform(sample.reshape(1, -1))[0]

        elif method_name == "gadf":
            img = gadf.fit_transform(sample.reshape(1, -1))[0]

        elif method_name == "rp":
            img = rp.fit_transform(sample.reshape(1, -1))[0]

        elif method_name == "mtf":
            img = mtf.fit_transform(sample.reshape(1, -1))[0]

        elif method_name == "spectrogram":
            save_path = os.path.join(base_dir, str(label), f"{i}.png")
            save_spectrogram(sample, save_path)
            continue

        else:
            continue

        save_path = os.path.join(base_dir, str(label), f"{i}.png")
        save_image(img, save_path)

        if i % 100 == 0:
            print(f"{i}/{len(X)} done")


def main():
    print("Loading dataset...")

    X_train = np.load(os.path.join(BASE_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(BASE_PATH, "y_train.npy"))
    X_test = np.load(os.path.join(BASE_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(BASE_PATH, "y_test.npy"))

    # methods = ["gaf", "gadf", "rp", "mtf", "spectrogram"]
    methods = ["gaf", "gadf", "rp", "spectrogram"]

    for method in methods:
        process_split(X_train, y_train, "train", method)
        process_split(X_test, y_test, "test", method)

    print("\nAll image datasets created!")


if __name__ == "__main__":
    main()
