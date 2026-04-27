import numpy as np
import os
from collections import Counter


INPUT_PATH = "datasets/Earthquakes"
OUTPUT_PATH = "datasets/Earthquakes/augmented"


def jitter(x, sigma=0.03):
    return x + np.random.normal(0, sigma, size=x.shape)


def scaling(x, sigma=0.15):
    factor = np.random.normal(1.0, sigma)
    return x * factor


def time_shift(x, shift_max=15):
    shift = np.random.randint(-shift_max, shift_max)

    if shift > 0:
        return np.concatenate([np.zeros(shift), x[:-shift]])
    elif shift < 0:
        return np.concatenate([x[-shift:], np.zeros(-shift)])
    return x


def time_warp(x, sigma=1.25, knot=4):
    orig_steps = np.arange(len(x))

    random_warps = np.random.normal(1.0, sigma, size=knot)
    warp_steps = np.linspace(0, len(x) - 1, num=knot)

    warped = np.interp(
        orig_steps,
        np.cumsum(np.interp(orig_steps, warp_steps, random_warps)),
        x
    )

    return warped


def light_augment(x):
    if np.random.rand() < 0.7:
        x = jitter(x)
    if np.random.rand() < 0.5:
        x = scaling(x)
    return x


def strong_augment(x):
    if np.random.rand() < 0.9:
        x = jitter(x)

    if np.random.rand() < 0.8:
        x = scaling(x)

    if np.random.rand() < 0.7:
        x = time_shift(x)

    if np.random.rand() < 0.6:
        x = time_warp(x)

    return x


def augment_train(X, y):
    print("\nAugmenting TRAIN data...")

    counts = Counter(y)
    print("Original distribution:", counts)

    minority_class = min(counts, key=counts.get)

    X_aug, y_aug = [], []

    for i in range(len(X)):
        sample = X[i]
        label = y[i]

        base = sample.squeeze()

        X_aug.append(sample)
        y_aug.append(label)

        if label == minority_class:
            for _ in range(6):
                aug = strong_augment(base.copy())
                X_aug.append(aug.reshape(sample.shape))
                y_aug.append(label)

        else:
            aug = light_augment(base.copy())
            X_aug.append(aug.reshape(sample.shape))
            y_aug.append(label)

    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)

    print("Augmented distribution:", Counter(y_aug))
    print("Train shape:", X_aug.shape)

    return X_aug, y_aug


def augment_test(X, y):
    print("\nTest data NOT augmented (good).")
    return X, y


def main():
    print("Loading dataset...")

    X_train = np.load(os.path.join(INPUT_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(INPUT_PATH, "y_train.npy"))
    X_test = np.load(os.path.join(INPUT_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(INPUT_PATH, "y_test.npy"))

    X_train_aug, y_train_aug = augment_train(X_train, y_train)
    X_test_aug, y_test_aug = augment_test(X_test, y_test)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train_aug)
    np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train_aug)
    np.save(os.path.join(OUTPUT_PATH, "X_test.npy"), X_test_aug)
    np.save(os.path.join(OUTPUT_PATH, "y_test.npy"), y_test_aug)

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
