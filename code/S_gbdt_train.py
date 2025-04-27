import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
def load_weighted_data(weighted_train_path, weighted_val_path):
    weighted_train = pd.read_csv(weighted_train_path)
    weighted_val = pd.read_csv(weighted_val_path)
    return weighted_train, weighted_val
def preprocess_data(train, val):
    X_train = train.drop(columns=['Target'])
    y_train = train['Target']
    X_val = val.drop(columns=['Target'])
    y_val = val['Target']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, y_train, X_val_scaled, y_val
def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    model = GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=7,
        min_samples_leaf=30,
        min_samples_split=30,
        n_estimators=300,
        subsample=0.8,
        random_state=42
    ) #the best parameters obtained by the grid search result

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    os.makedirs(r'', exist_ok=True)
    unique_labels = sorted(np.unique(y_val))
    cm_df = pd.DataFrame(cm,
                         index=[f'True_{label}' for label in unique_labels],
                         columns=[f'Pred_{label}' for label in unique_labels])
    cm_df.to_csv(r'')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels)
    plt.title('GBDT Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(r'')
    plt.close()
    return accuracy, model

def main():
    weighted_train_path = r''
    weighted_val_path = r''
    weighted_train, weighted_val = load_weighted_data(
        weighted_train_path, weighted_val_path
    )
    X_train_weighted, y_train_weighted, X_val_weighted, y_val_weighted = preprocess_data(
        weighted_train, weighted_val
    )

    accuracy, model = train_and_evaluate_model(
        X_train_weighted, y_train_weighted,
        X_val_weighted, y_val_weighted
    )


if __name__ == "__main__":
    main()