# Time Series → Image Classification Pipeline

This repository provides an end-to-end pipeline for time-series classification using both traditional ML models and CNNs on image representations of the data.

---

## Project Structure

### 1. `load_dataset.py`
- Loads datasets from the UCR/UEA archive
- Saves them locally as `.npy` files
- Prints dataset stats and generates a sample plot

### 2. `augment_dataset.py`
- Performs data augmentation on training data
- Uses jittering, scaling, time shifting, and time warping
- Balances class distribution
- Saves augmented dataset

### 3. `create_image_dataset.py`
- Converts time-series into images:
  - GAF
  - GADF
  - RP
  - Combo (RGB)
- Saves structured image dataset

### 4. `traditional_models.py`
- Runs:
  - DTW KNN
  - Euclidean KNN
  - Random Forest
  - SVM
- Saves results to text files

### 5. `train_cnn.py`
- Trains CNN on image datasets
- Supports multiple representations
- Handles class imbalance
- Saves logs, models, and summaries

---

## How to Use

### Clone the repo
```
git clone https://github.com/asmit-mit/time-series-image-classification.git
cd time-series-image-classification
```

### Create virtual environment
```
python -m venv .venv
source .venv/bin/activate
```

### Install requirements
```
pip install -r requirements.txt
```

### Run everything
```
python run_all.py --dataset <dataset-name> --epochs <training-epochs>
```

---

## Arguments

- `--dataset` → Dataset name (default: Earthquakes)
- `--epochs` → CNN training epochs (default: 80)

---

## Outputs

- Augmented data → `datasets/<dataset-name>/augmented`
- Images → `datasets/<dataset-name>/augmented/images`
- Traditional results → `train/<dataset-name>/traditional_models`
- CNN outputs → `train/<dataset-name>/cnn`
