# ================================================
# dsp_csv_project.py — Fully Adaptable Version
# ================================================

import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import librosa

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from xgboost import XGBClassifier
from audio_prep import extract_features

# ====================================================
# === 1. LOAD & PREPARE TRAINING DATA
# ====================================================
print("📥 Loading training data...")
df = pd.read_csv(r"D:\vscode projects\ml folder dsp project\irmas_train_spec.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

# Encode labels
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# ====================================================
# === 2. FEATURE SCALING
# ====================================================
print("⚙️ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====================================================
# === 3. RANDOM FOREST HYPERPARAMETER TUNING
# ====================================================
print("🔍 Running GridSearchCV for Random Forest...")
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 40],
    "min_samples_split": [2, 5, 10],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_scaled, y)
print("✅ Best Random Forest params:", grid.best_params_)
best_rf = grid.best_estimator_

# ====================================================
# === 4. DIMENSIONALITY REDUCTION / FEATURE SELECTION
# ====================================================
print("🧩 Applying PCA and SelectKBest...")

num_features = X_scaled.shape[1]
n_components = min(20, num_features)

datasets = {
    "Original": X_scaled
}

if num_features > 1:
    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    datasets["PCA"] = X_pca

    # SelectKBest
    selector = SelectKBest(score_func=f_classif, k=n_components)
    X_kbest = selector.fit_transform(X_scaled, y)
    datasets["KBest"] = X_kbest
else:
    print(f"⚠️ Skipping PCA and SelectKBest — only {num_features} feature(s) available.")

# ====================================================
# === 5. MODEL DEFINITIONS
# ====================================================
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": xgb_model,
    "Tuned RF": best_rf,
    "SVM (RBF)": SVC(C=10, gamma='scale', kernel='rbf', probability=True),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05),
    "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.1),
    "XGBoost-Boosted": XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=8)
}

# ====================================================
# === 6. TEST FEATURE EXTRACTION
# ====================================================
def extract_test_data(folder):
    test_data, filenames = [], []
    for file in glob.glob(os.path.join(folder, "*.wav")):
        try:
            features = extract_features(file)
            if features is not None:
                test_data.append(features)
                filenames.append(os.path.basename(file))
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
    return np.array(test_data), filenames

# ====================================================
# === 7. LOAD TEST DATA
# ====================================================
test_folder = r"D:\download-cdrive\IRMAS-TestingData-Part3\IRMAS-TestingData-Part3\Part3"
print(f"📂 Loading test data from {test_folder} ...")
X_test_raw, test_files = extract_test_data(test_folder)

print(f"Train features: {X_scaled.shape[1]}")
print(f"Test features: {X_test_raw.shape[1]}")

if X_test_raw.shape[1] != X_scaled.shape[1]:
    print("⚠️ Feature count mismatch! Adjusting...")
    X_test_raw = X_test_raw[:, :X_scaled.shape[1]]

X_test = scaler.transform(X_test_raw)

# ====================================================
# === 8. FINAL MODEL PREDICTION ON TEST DATA
# ====================================================
final_model = best_rf  # Or models["XGBoost"]
print("🏁 Training final model and predicting test data...")

final_model.fit(X_scaled, y)
y_test_pred = final_model.predict(X_test)
predicted_labels = y_encoder.inverse_transform(y_test_pred)

pd.DataFrame({
    "filename": test_files,
    "predicted_label": predicted_labels
}).to_csv("test_predictions_mfcc.csv", index=False)
print("✅ Test predictions saved to test_predictions_mfcc.csv")

# ====================================================
# === 9. CROSS-VALIDATION EVALUATION
# ====================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_rf, X_scaled, y, cv=cv, scoring="accuracy")
print(f"📊 Cross-validated accuracy: {np.mean(scores):.4f}")

# ====================================================
# === 10. MODEL EVALUATION ON TRAIN/VAL SPLIT
# ====================================================
print("🔎 Evaluating models across datasets...")
all_results = []

for dname, X_data in datasets.items():
    print(f"\n=== Dataset: {dname} ===")
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y, test_size=0.2, stratify=y, random_state=42
    )

    for name, model in models.items():
        print(f"\n--- {name} ({dname}) ---")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        report = classification_report(
            y_val, y_pred, target_names=y_encoder.classes_, output_dict=True
        )
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f"{name}_{dname}_classification_report_spec.csv")

        print(df_report)

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=y_encoder.classes_,
            yticklabels=y_encoder.classes_,
        )
        plt.title(f"{name} - Confusion Matrix ({dname})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{name}_{dname}_confusion_matrix_spec.png")
        plt.close()

        # Collect results
        all_results.append({
            "Dataset": dname,
            "Model": name,
            "Accuracy": report["accuracy"],
            "Macro Avg F1": report["macro avg"]["f1-score"],
            "Weighted Avg F1": report["weighted avg"]["f1-score"],
        })

# ====================================================
# === 11. SAVE SUMMARY RESULTS
# ====================================================
pd.DataFrame(all_results).to_csv("summary_results_spec.csv", index=False)
print("\n✅ Summary saved to summary_results_spec.csv")
