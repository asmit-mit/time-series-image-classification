import numpy as np
import os
from pyts.image import GramianAngularField, RecurrencePlot
from scipy.signal import resample
from PIL import Image

BASE_PATH = "datasets/Earthquakes/augmented"
IMAGE_SIZE = 128


def normalize_to_uint8(img):
    img_min, img_max = img.min(), img.max()
    if img_max - img_min == 0:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - img_min) / (img_max - img_min)
    return (img * 255).astype(np.uint8)


def save_single_channel(img, path):
    img = normalize_to_uint8(img)
    Image.fromarray(img).save(path)


def save_combo(gaf, gadf, rp, path):
    gaf = normalize_to_uint8(gaf)
    gadf = normalize_to_uint8(gadf)
    rp = normalize_to_uint8(rp)

    combo = np.stack([gaf, gadf, rp], axis=-1)
    Image.fromarray(combo).save(path)


def process_split(X, y, split_name):
    print(f"\nProcessing {split_name}")

    X = X.squeeze()

    methods = ["gaf", "gadf", "rp", "combo"]

    for method in methods:
        for label in np.unique(y):
            os.makedirs(
                os.path.join(BASE_PATH, "images", method, split_name, str(label)),
                exist_ok=True
            )

    gaf = GramianAngularField(image_size=IMAGE_SIZE, method='summation')
    gadf = GramianAngularField(image_size=IMAGE_SIZE, method='difference')
    rp = RecurrencePlot()

    for i in range(len(X)):
        sample = X[i]
        label = y[i]

        sample_resampled = resample(sample, IMAGE_SIZE)
        sample_reshaped = sample_resampled.reshape(1, -1)

        gaf_img = gaf.fit_transform(sample_reshaped)[0]
        gadf_img = gadf.fit_transform(sample_reshaped)[0]
        rp_img = rp.fit_transform(sample_reshaped)[0]

        base = os.path.join(BASE_PATH, "images")

        gaf_path = os.path.join(base, "gaf", split_name, str(label), f"{i}.png")
        gadf_path = os.path.join(base, "gadf", split_name, str(label), f"{i}.png")
        rp_path = os.path.join(base, "rp", split_name, str(label), f"{i}.png")
        combo_path = os.path.join(base, "combo", split_name, str(label), f"{i}.png")

        save_single_channel(gaf_img, gaf_path)
        save_single_channel(gadf_img, gadf_path)
        save_single_channel(rp_img, rp_path)
        save_combo(gaf_img, gadf_img, rp_img, combo_path)

        if i % 100 == 0:
            print(f"{i}/{len(X)} done")


def main():
    print("Loading dataset...")

    X_train = np.load(os.path.join(BASE_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(BASE_PATH, "y_train.npy"))
    X_test = np.load(os.path.join(BASE_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(BASE_PATH, "y_test.npy"))

    process_split(X_train, y_train, "train")
    process_split(X_test, y_test, "test")

    print("\nAll image datasets created!")


if __name__ == "__main__":
    main()
