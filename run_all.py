import os
import argparse

import load_dataset
import augment_dataset
import create_image_dataset
import traditional_models
import train_cnn


def update_paths(dataset_name):
    """
    Dynamically update paths in all modules
    """

    base_dataset_path = f"datasets/{dataset_name}"
    augmented_path = f"{base_dataset_path}/augmented"
    image_path = f"{augmented_path}/images"

    load_dataset.BASE_DATASET_PATH = "datasets"

    augment_dataset.INPUT_PATH = base_dataset_path
    augment_dataset.OUTPUT_PATH = augmented_path

    create_image_dataset.BASE_PATH = augmented_path

    traditional_models.BASE_PATH = augmented_path
    traditional_models.OUTPUT_DIR = f"train/{dataset_name}/traditional_models"

    train_cnn.BASE_PATH = image_path
    train_cnn.OUTPUT_DIR = f"train/{dataset_name}/cnn"


def run_pipeline(dataset_name):
    print(f"\n========== RUNNING PIPELINE FOR: {dataset_name} ==========\n")

    update_paths(dataset_name)

    print("\n[1] Loading dataset...")
    load_dataset.main()

    print("\n[2] Augmenting dataset...")
    augment_dataset.main()

    print("\n[3] Creating image dataset...")
    create_image_dataset.main()

    print("\n[4] Running traditional models...")
    traditional_models.main()

    print("\n[5] Training CNN...")
    train_cnn.main()

    print("\n========== DONE ==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Earthquakes",
        help="Dataset name (UCR format or existing folder)"
    )

    args = parser.parse_args()

    run_pipeline(args.dataset)
