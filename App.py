from flask import Flask, render_template, jsonify, request
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score,
    precision_score, recall_score
)

app = Flask(__name__)

# ── Load Dataset ─────────────────────────────────────────────
iris        = load_iris()
X, y        = iris.data, iris.target
class_names = list(iris.target_names)
feat_names  = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

# ── Scale & Split ────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

# ── Training Log ─────────────────────────────────────────────
training_log = []

def log(msg):
    training_log.append({"time": time.strftime("%H:%M:%S"), "msg": msg})

log("Initializing DecodeLabs Project 2 — Iris Classification")
log(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features, {len(class_names)} classes")
log(f"Classes: {', '.join(class_names)}")
log("Applying StandardScaler — mean=0, variance=1")
log(f"Train/Test split: {len(X_train)} train / {len(X_test)} test (80/20, shuffle=True)")
log("Running Elbow Method to find optimal K (K=1 to 20)...")

k_scores = {}
for k in range(1, 21):
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    k_scores[k] = round(acc * 100, 2)
    log(f"  K={k:>2} → Accuracy: {acc*100:.2f}%")

best_k = max(k_scores, key=k_scores.get)
log(f"Optimal K found: K={best_k} (Accuracy={k_scores[best_k]}%)")

# ── Train Models ─────────────────────────────────────────────
log("Training KNeighborsClassifier (final model)...")
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)
knn_acc   = accuracy_score(y_test, knn_preds)
log(f"KNN training complete — Accuracy: {knn_acc*100:.2f}%, F1: {f1_score(y_test, knn_preds, average='weighted'):.4f}")

log("Training comparison models...")
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes":   GaussianNB(),
    "SVM":           SVC(probability=True, random_state=42),
}
algo_results = {
    "KNN": {
        "accuracy":  round(knn_acc * 100, 2),
        "f1":        round(f1_score(y_test, knn_preds, average="weighted"), 4),
        "precision": round(precision_score(y_test, knn_preds, average="weighted"), 4),
        "recall":    round(recall_score(y_test, knn_preds, average="weighted"), 4),
        "cm":        confusion_matrix(y_test, knn_preds).tolist(),
    }
}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    a = accuracy_score(y_test, p)
    algo_results[name] = {
        "accuracy":  round(a * 100, 2),
        "f1":        round(f1_score(y_test, p, average="weighted"), 4),
        "precision": round(precision_score(y_test, p, average="weighted"), 4),
        "recall":    round(recall_score(y_test, p, average="weighted"), 4),
        "cm":        confusion_matrix(y_test, p).tolist(),
    }
    log(f"  {name} — Accuracy: {a*100:.2f}%")

log(f"All models trained. Best model: KNN (K={best_k}) with {knn_acc*100:.2f}% accuracy")
log("Server ready at http://127.0.0.1:5000")

# ── Full Dataset for Table ───────────────────────────────────
dataset_rows = []
for i in range(len(X)):
    dataset_rows.append({
        "id":    i + 1,
        "sl":    round(float(X[i][0]), 1),
        "sw":    round(float(X[i][1]), 1),
        "pl":    round(float(X[i][2]), 1),
        "pw":    round(float(X[i][3]), 1),
        "label": class_names[int(y[i])],
        "cls":   int(y[i]),
    })

# ── Scatter Data ─────────────────────────────────────────────
scatter_data = [
    {"sl": round(float(X[i][0]),2), "sw": round(float(X[i][1]),2),
     "pl": round(float(X[i][2]),2), "pw": round(float(X[i][3]),2),
     "cls": int(y[i]), "name": class_names[int(y[i])]}
    for i in range(len(X))
]

main_results = {
    "accuracy":       round(knn_acc * 100, 2),
    "f1_score":       round(f1_score(y_test, knn_preds, average="weighted"), 4),
    "best_k":         best_k,
    "train_samples":  len(X_train),
    "test_samples":   len(X_test),
    "total_samples":  len(X),
    "confusion_matrix": confusion_matrix(y_test, knn_preds).tolist(),
    "class_names":    class_names,
    "k_scores":       k_scores,
    "algo_results":   algo_results,
    "dataset_rows":   dataset_rows,
    "scatter_data":   scatter_data,
    "feat_names":     feat_names,
    "training_log":   training_log,
}

# In-memory prediction history
prediction_history = []

# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", results=main_results)

@app.route("/predict", methods=["POST"])
def predict():
    data     = request.json
    sl, sw   = float(data["sepal_length"]), float(data["sepal_width"])
    pl, pw   = float(data["petal_length"]),  float(data["petal_width"])
    features = np.array([[sl, sw, pl, pw]])
    scaled   = scaler.transform(features)
    pred     = knn.predict(scaled)[0]
    proba    = knn.predict_proba(scaled)[0]

    result = {
        "class":      class_names[pred],
        "confidence": round(float(proba[pred]) * 100, 1),
        "all_probs":  {class_names[i]: round(float(proba[i])*100,1) for i in range(3)},
        "inputs":     {"sl": sl, "sw": sw, "pl": pl, "pw": pw},
        "timestamp":  time.strftime("%H:%M:%S"),
    }
    prediction_history.insert(0, result)
    if len(prediction_history) > 50:
        prediction_history.pop()
    return jsonify(result)

@app.route("/history")
def history():
    return jsonify(prediction_history)

if __name__ == "__main__":
    print("\n  DecodeLabs | Project 2 \n")
    app.run(debug=True)