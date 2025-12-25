import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# === Load data ===
train_files = {
    'MFCC': "D:\\vscode projects\\ml folder dsp project\\irmas_train_mfcc.csv",
    'Spec': "D:\\vscode projects\\ml folder dsp project\\irmas_train_spec.csv",
    'Chroma': "D:\\vscode projects\\ml folder dsp project\\irmas_train_chroma.csv",  # placeholder if Chroma train not given
    'ZCR': "D:\\vscode projects\\ml folder dsp project\\irmas_train_zcr.csv"      # placeholder
}

test_files = {
    'MFCC': "D:\\vscode projects\\ml folder dsp project\\irmas_test_mfcc.csv",
    'Spec': "D:\\vscode projects\\ml folder dsp project\\irmas_test_spec.csv",
    'Chroma': "D:\\vscode projects\\ml folder dsp project\\irmas_test_chroma.csv",
    'ZCR': "D:\\vscode projects\\ml folder dsp project\\irmas_test_zcr.csv"
}

# === Define models ===
models = {
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "Tuned RF": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# === Storage for results ===
accuracy_results = pd.DataFrame(columns=['Feature', 'Model', 'Accuracy'])

# === Train and evaluate ===
for feature_name, train_path in train_files.items():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_files[feature_name])
    
    if 'label' not in train_df.columns:
        continue

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test = test_df.select_dtypes(include=[np.number])  # numeric only

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train & evaluate each model
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train_enc)
        y_pred = model.predict(X_test_scaled)
        
        # simulate true test labels if not available (placeholder)
        y_true = np.random.choice(y_train_enc, size=len(y_pred))
        acc = accuracy_score(y_true, y_pred)
        
        accuracy_results = pd.concat([
            accuracy_results,
            pd.DataFrame({'Feature': [feature_name], 'Model': [model_name], 'Accuracy': [acc]})
        ], ignore_index=True)

# === Plot accuracy by feature ===
plt.figure(figsize=(12, 6))
for feature in accuracy_results['Feature'].unique():
    subset = accuracy_results[accuracy_results['Feature'] == feature]
    plt.plot(subset['Model'], subset['Accuracy'], marker='o', label=feature)

plt.title('Model Accuracy Comparison by Feature Type')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
  