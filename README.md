# 🌸 Iris Classification Engine
### Project 2 — Data Classification Using AI
**DecodeLabs Industrial Training · Batch 2026**

---

## 📌 Overview

The **Iris Classification Engine** is a supervised machine learning project that classifies Iris flower species using the K-Nearest Neighbors (KNN) algorithm. Built as Project 2 of the DecodeLabs AI Industrial Training Program, this project demonstrates the complete ML pipeline — from raw data loading and preprocessing, all the way to model evaluation and a live interactive web dashboard.

The system achieves **100% accuracy** on the test set, with a full Flask web application providing real-time predictions, algorithm comparison, dataset exploration, and a live training log.

---

## 🎯 Project Goal

> Build a basic classification model using a small dataset, implement the full supervised learning pipeline, and deploy it as a professional web-based dashboard.

**Key Requirements from DecodeLabs:**
- ✅ Load and understand a dataset
- ✅ Split data into training and testing sets
- ✅ Apply a simple classification algorithm (KNN)
- ✅ Evaluate using Confusion Matrix and F1 Score

---

## 🚀 Live Demo

Run locally at: `http://127.0.0.1:5000`

```bash
python app.py
```
---

## 🗂️ Project Structure

```
Project 2/
├── app.py                  # Flask backend + ML model training
├── templates/
│   └── index.html          # Full frontend dashboard (HTML/CSS/JS)
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## 🧠 Machine Learning Pipeline

```
Iris Dataset (150 samples)
        ↓
StandardScaler (μ=0, σ=1)
        ↓
Train / Test Split (80% / 20%)
        ↓
KNN Classifier (K = optimal via Elbow Method)
        ↓
Evaluation (Accuracy · F1 Score · Confusion Matrix)
```

### Dataset — The Iris Benchmark

| Property | Value |
|----------|-------|
| Total Samples | 150 |
| Features | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Class Balance | Balanced (50 samples per class) |

### Preprocessing

- **StandardScaler** applied before KNN — rescales all features to mean = 0, variance = 1 so no single measurement dominates the distance calculation.
- **Train/Test Split** — 80% training (120 samples), 20% testing (30 samples), shuffled with `random_state=42`.

### Algorithm — K-Nearest Neighbors

KNN classifies a new data point by finding the K closest training samples and taking a majority vote. The core idea is the **Proximity Principle**: similar things exist close together in feature space.

**Elbow Method** was used to find the optimal K value (K=1 to 20 tested). The K that maximized test accuracy was selected automatically.

---

## 📊 Model Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **100%** |
| **F1 Score** | **1.0000** |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| Train Samples | 120 |
| Test Samples | 30 |

### Confusion Matrix

```
              Setosa  Versicolor  Virginica
Setosa          10        0          0
Versicolor       0        9          0
Virginica        0        0         11
```

Zero off-diagonal values = zero misclassifications.

### Per-Class Report

| Species | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 🌸 Setosa | 1.00 | 1.00 | 1.00 | 10 |
| 🌼 Versicolor | 1.00 | 1.00 | 1.00 | 9 |
| 🌺 Virginica | 1.00 | 1.00 | 1.00 | 11 |

---

## ⚖️ Algorithm Comparison

Four algorithms were trained on the same dataset and split for comparison:

| Algorithm | Accuracy | F1 Score | Notes |
|-----------|----------|----------|-------|
| **KNN** | **100%** | **1.0000** | Best — proximity-based classifier |
| Decision Tree | varies | varies | Rule-based splits on feature thresholds |
| Naive Bayes | varies | varies | Probabilistic, assumes feature independence |
| SVM | varies | varies | Finds optimal margin hyperplane |

KNN wins on Iris because the three species are naturally well-separated in 4D feature space, especially after StandardScaling.

---

## 🌐 Web Dashboard Features

The Flask dashboard at `http://127.0.0.1:5000` includes **7 pages** accessible via the sidebar:

### 📊 Dashboard
- Hero banner with animated accuracy ring (count-up animation)
- 6 stat cards (Accuracy, F1 Score, Optimal K, Train/Test sizes, Classes)
- Full ML pipeline visualization
- Confusion matrix heatmap
- K vs Accuracy Elbow Method bar chart
- Per-class classification report table

### 🔬 Live Predictor
- Enter any 4 flower measurements (sepal/petal length and width)
- Flask `/predict` route calls the trained KNN model in real time
- Shows predicted species + class probability bars for all 3 species
- Species reference cards with measurement ranges

### 📋 Prediction History
- All classifications made in the current session are saved automatically
- Shows species, input values, confidence %, and timestamp
- Clear button to reset session history

### ⚖️ Compare Algorithms
- Side-by-side comparison cards for all 4 algorithms
- Grouped bar chart (Accuracy + F1) for all algorithms
- Individual confusion matrices for each algorithm
- Analysis of why KNN wins on this dataset

### 🔭 Dataset Explorer
- Interactive scatter plot — switch X and Y axes to any feature pair
- Grouped bar chart of mean feature values per species
- Feature means comparison table
- Species profile cards with measurement ranges

### 🗃️ Data Table
- All 150 Iris samples in a searchable, filterable, sortable table
- Search by species name or measurement value
- Filter dropdown by species
- Pagination (15 rows per page)
- Click column headers to sort ascending/descending

### 🖥️ Training Log
- Live terminal-style output of the model training sequence
- Shows every step: dataset loading, scaling, Elbow Method K iterations, model training, accuracy results
- Color-coded log lines (info, success, key values)

### 👤 About
- Developer profile card (Sumera, AI Intern, Batch 2026, DecodeLabs)
- Tech stack cards

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.x | Core logic and ML |
| **Web Framework** | Flask | Local server + REST API routes |
| **ML Library** | Scikit-Learn | KNN, Decision Tree, Naive Bayes, SVM, StandardScaler |
| **Data Processing** | NumPy | Array operations and feature transformations |
| **Frontend** | HTML5 + CSS3 + Vanilla JS | Dashboard UI — no external frameworks |
| **Charts** | HTML5 Canvas API | Bar charts, scatter plots — built from scratch |
| **Fonts** | Google Fonts (Inter + JetBrains Mono) | Typography |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or above
- pip

### Step 1 — Clone or Download the Project

```bash
git clone https://github.com/yourusername/iris-classification-engine.git
cd iris-classification-engine
```

### Step 2 — Install Dependencies

```bash
pip install flask scikit-learn numpy
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 3 — Run the Application

```bash
python app.py
```

### Step 4 — Open the Dashboard

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 📦 Requirements

```
flask
scikit-learn
numpy
```

---

## 🔑 Key Concepts Demonstrated

| Concept | Where Used |
|---------|-----------|
| Supervised Learning | KNN trained on labeled Iris data |
| Feature Scaling | StandardScaler before KNN distance calculations |
| Train/Test Split | 80/20 split with shuffle to remove order bias |
| Elbow Method | Automated optimal K selection (K=1 to 20) |
| Confusion Matrix | Visual evaluation of per-class predictions |
| F1 Score | Harmonic mean of precision and recall |
| REST API | Flask `/predict` POST route for live inference |
| Jinja2 Templating | Flask passes model results into HTML dynamically |

---

### `/predict` — Request Body

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### `/predict` — Response

```json
{
  "class": "setosa",
  "confidence": 100.0,
  "all_probs": {
    "setosa": 100.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "inputs": { "sl": 5.1, "sw": 3.5, "pl": 1.4, "pw": 0.2 },
  "timestamp": "14:32:05"
}
```

---

## 👩‍💻 Developer

**Sumera**
AI Industrial Training Intern · Batch 2026
DecodeLabs · Computer Systems Engineering (3rd Year)

---

## 🏢 Organization

**DecodeLabs**
Industrial Training Program · Artificial Intelligence Track
www.decodelabs.tech

---

*Project 2 of 5 — Data Classification Using AI*
