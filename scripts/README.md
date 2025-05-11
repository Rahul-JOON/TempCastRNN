# 📂 scripts/

This folder contains the **core scripts** responsible for training and evaluating the transformer-based temperature forecasting model.

### 1. `train.py` – Model Training & Definition

This script contains all the components necessary to train the transformer model.

#### 🔧 Key Components:

- **`ForecastDataset`**  
  A custom PyTorch `Dataset` class to wrap the input matrices and target temperatures.

- **`ForecastTransformer`**  
  A transformer-based neural network for processing 12×12 forecast matrices.  
  Architecture:
  - Input: (12, 12) matrix (forecast timeline)
  - Embedding: Linear(12 → d_model)
  - Encoder: 2-layer Transformer Encoder (4 heads)
  - Output: Final temperature prediction for 12th hour

- **`generate_square_subsequent_mask()`**  
  Creates a mask for causal (autoregressive-style) transformer attention, ensuring future time steps don't influence predictions of the past.

- **`train_model()`**  
  Orchestrates the training process:
  - Trains the model on training data using `SmoothL1Loss`
  - Validates periodically
  - Saves the best model (`best_model.pth`) and corresponding metadata (`best_model.json`)
  - Early-stops when validation loss goes below a defined threshold
  - Logs training/validation loss to CSV and PNG (`logs/train_logs/<timestamp>/`)

---

### 2. `evaluate.py` – Model Evaluation

This script is used for evaluating the trained model on new datasets, given any city and time window.

#### 🔍 Major Functions:

- **`download_csv()`**  
  Downloads CSV forecast data using a REST API (`/download`) for the specified city and time window.

- **`load_model()`**  
  Loads a previously trained model from a `.pth` file.

- **`prepare_inference_input()`**  
  Converts a 12-hour forecast vector into a 12×12 matrix by duplicating and adding synthetic decay/noise to simulate older predictions.

- **`evaluate_model()`**  
  - Processes the dataset
  - Generates predictions using the model
  - Matches predictions with the most accurate actual temperatures
  - Computes and logs:
    - **MAE (Mean Absolute Error)**
    - **RMSE (Root Mean Squared Error)**
  - Saves:
    - Evaluation plot (`logs/eval_logs/forecast_plot_<timestamp>.png`)
    - Evaluation results (`logs/eval_logs/eval_results_<timestamp>.json`)

- **`evaluate()`**  
  High-level wrapper for evaluating across **multiple cities**. Automatically downloads and aggregates data, then runs evaluation using the combined dataset.

---

## 🔄 Integration with `model.py`

Both `train.py` and `evaluate.py` are imported and called from the root-level `model.py` script, which acts as the central CLI interface.

---
