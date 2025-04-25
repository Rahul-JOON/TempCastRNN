<div align="center">

# 📡 TempCastRNN

  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/PyTorch-1.13%2B-red" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <a href="https://github.com/Rahul-JOON/Forecast-Journal">
    <img src="https://img.shields.io/badge/Linked%20Repo-Forecast%20Journal-0a8" />
  </a>

**Hourly Forecast Refinement Using Temporal Attention**

</div>

---

### 📖 Project Overview
**TempCastRNN** is a transformer-based time series model that learns the temporal evolution of hourly weather forecasts to produce more accurate predictions for a target hour. It is designed to work seamlessly with the [Forecast Journal](https://github.com/Rahul-JOON/Forecast-Journal) repository, using its data to enhance predictive accuracy.

This repository implements the ML modeling component described in the Forecast Journal's roadmap. It takes historical 12×12 forecast matrices and predicts the accurate temperature at a 12-hour horizon using temporal attention.

---

### 🔍 Motivation
Weather forecasts update hourly, and those updates contain patterns. Instead of using only the latest forecast, this model leverages a sequence of previous hourly forecasts to learn how prediction patterns evolve.

The aim is to correct inaccuracies in the 12-hour forecast by recognizing trends in past forecast changes.

---

### 📚 Key Concepts

#### 🔁 Recurrent Neural Networks (RNNs)
RNNs process sequences one timestep at a time and maintain memory of previous steps. While useful for sequential modeling, they struggle with long-range dependencies.

#### ✨ Transformer & Attention
Transformers use self-attention to focus on important parts of the input sequence regardless of position. This allows the model to capture dependencies across the entire 12-hour forecasting sequence without vanishing gradients.

---

### ⚙️ Implementation

#### 🧠 Core Idea
Each hour, the Forecast Journal repository records forecasts made for the next 12 hours. Stacking these for 12 hours forms a 12×12 matrix:

```
Hour 0 → [T0, T1, ..., T11]
Hour 1 → [T1, T2, ..., T12]
...
Hour 11 → [T11, ..., T22]
```

This matrix is used to predict the true value of temperature at Hour 12.

#### 🏗️ Model Architecture
- **Input**: 12×12 forecast matrix
- **Embedding**: Linear layer to encode each forecast vector
- **Transformer Encoder**: 2 layers with multi-head attention
- **Final Dense Layer**: Outputs a single scalar (predicted temperature)

```
[12×12 Matrix] → Embedding → Transformer Encoder → Output (1 value)
```

#### 🚫 Masking Strategy
A causal mask is applied so each row (forecast made at hour `i`) only attends to itself and prior rows. This ensures that predictions are not influenced by future data.

#### 🔁 Data Download Automation
The model automatically fetches training data by triggering a download request from the Forecast Journal frontend. It supports city-based filtering and dynamic date range selection directly from configuration:

```bash
python model.py --city "Delhi" --start "2025-03-01" --end "2025-03-30"
```

Internally, this script:
- Sends an HTTP POST request to the dashboard's download endpoint.
- Parses and saves the returned CSV.
- Transforms it into a 12×12 tensor dataset.

---

### 🧪 Training
To begin training:
```bash
python scripts/train.py
```

- **Loss**: Mean Absolute Error (MAE)
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing
- **Logging**: Epoch-wise training logs printed and saved

Training data is pulled from preprocessed CSV files downloaded via the Forecast Journal interface.

---

### 📊 Prediction
To generate predictions from a 12×1 forecast vector (e.g. live input):
```bash
python scripts/predict_and_save.py
```

Predictions are exported to:
```
predictions/output_predictions.csv
```

New predictions are also displayed on the [Forecast Journal dashboard](https://forecast-journal.vercel.app/).

---

### 🔁 Integration with Forecast Journal
This repository is tightly integrated with the [Forecast Journal](https://github.com/Rahul-JOON/Forecast-Journal) project:

- **Source**: Hourly weather forecasts stored in NeonDB
- **Frontend**: Public dashboard for export and visualization
- **Automation**: This model triggers CSV downloads from the hosted dashboard programmatically using city/date filters, then parses and reshapes the data for model input.

---

### 📈 Visualization
Loss curves are generated post-training to help evaluate convergence:

![Training Loss](docs/loss_curve_example.png)

---

### 📓 Jupyter Notebook Demo
A demonstration notebook is available under:
```
notebooks/TempCastRNN_Demo.ipynb
```
Steps included:
1. Download data with HTTP request
2. Transform into model format
3. Initialize and train TempCastRNN
4. Predict and export results
5. Visualize training loss

---

### 🙌 Contributions & Feedback
Feedback and contributions are encouraged. Open an issue or submit a pull request to improve the model, code, or documentation.

> Built for real-world weather prediction correction using time-aware deep learning.