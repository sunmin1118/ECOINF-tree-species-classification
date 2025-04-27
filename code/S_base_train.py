import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

class ShadowDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class EnhancedAttentionCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnhancedAttentionCNN, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.Softmax(dim=1)
        )

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        with torch.no_grad():
            test_input = torch.zeros(1, 1, input_size)
            feature_size = self._calculate_feature_size(test_input)

        self.classifier = nn.Sequential(
            nn.Linear(256 * feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _calculate_feature_size(self, x):
        return self.features(x).size(2)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        attention_weights = self.attention(x_flat)
        x_attended = x_flat * attention_weights
        x_conv = x_attended.view(x.size(0), 1, -1)
        features = self.features(x_conv)
        x = features.view(features.size(0), -1)
        output = self.classifier(x)
        return output, attention_weights


def train_and_evaluate_models(train_path, val_path):
    os.makedirs(r'', exist_ok=True)
    os.makedirs(r'', exist_ok=True)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    label_encoder = LabelEncoder()
    all_targets = pd.concat([train_df['Target'], val_df['Target']])
    label_encoder.fit(all_targets)
    train_df['Target_Encoded'] = label_encoder.transform(train_df['Target'])
    val_df['Target_Encoded'] = label_encoder.transform(val_df['Target'])
    joblib.dump(label_encoder, r'')
    X_train = train_df.drop(['Target', 'Target_Encoded'], axis=1).values
    y_train = train_df['Target_Encoded'].values
    X_val = val_df.drop(['Target', 'Target_Encoded'], axis=1).values
    y_val = val_df['Target_Encoded'].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, r'')

    models = {
        'CNN': None,
        'MLP': None,
        'Random Forest': None,
        'SVM': None
    }
    input_size = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_train))
    cnn_model = EnhancedAttentionCNN(input_size, num_classes)

    train_dataset = ShadowDataset(np.expand_dims(X_train_scaled, axis=1), y_train)
    val_dataset = ShadowDataset(np.expand_dims(X_val_scaled, axis=1), y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    num_epochs = 100
    best_cnn_accuracy = 0

    for epoch in range(num_epochs):
        cnn_model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs, _ = cnn_model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        cnn_model.eval()
        correct = 0
        total = 0
        y_pred_list = []
        y_true_list = []
        attention_weights_list = []

        with torch.no_grad():
            for val_features, val_labels in val_loader:
                outputs, attention_weights = cnn_model(val_features)
                _, predicted = torch.max(outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
                y_pred_list.extend(predicted.numpy())
                y_true_list.extend(val_labels.numpy())
                attention_weights_list.extend(attention_weights.numpy())

        cnn_accuracy = 100 * correct / total
        scheduler.step(cnn_accuracy)

        if cnn_accuracy > best_cnn_accuracy:
            best_cnn_accuracy = cnn_accuracy
            torch.save(cnn_model.state_dict(), r'')

            attention_df = pd.DataFrame(attention_weights_list,
                                        columns=[f'Feature_{i}' for i in range(attention_weights_list[0].shape[0])])
            attention_df.to_csv(r'', index=False)

    mlp_params = {
        'hidden_layer_sizes': [(50, 50), (100,), (50, 100, 50),'adjusted according to dataset'],
        'activation': ['relu','adjusted according to dataset'],
        'solver': ['adam','adjusted according to dataset'],
        'max_iter': [1000,'adjusted according to dataset']
    }
    mlp = MLPClassifier(random_state=42)
    mlp_grid = GridSearchCV(mlp, mlp_params, cv=5, scoring='f1_weighted')
    mlp_grid.fit(X_train_scaled, y_train)
    models['MLP'] = mlp_grid.best_estimator_

    joblib.dump(mlp_grid.best_estimator_, r'')

    rf_params = {
        'n_estimators': [100, 120, 140, 160, 180, 200, 'adjusted according to dataset'],
        'max_depth': [None, 10, 20, 30, 40, 50,'adjusted according to dataset'],
        'min_samples_split': [2, 3, 4, 5, 6, 'adjusted according to dataset']
    }
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1_weighted')
    rf_grid.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf_grid.best_estimator_
    joblib.dump(rf_grid.best_estimator_, r'')
    svm_params = {
        'C': [0.1, 0.5, 1, 5, 10,'adjusted according to dataset'],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svm = SVC(random_state=42, probability=True)
    svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='f1_weighted')
    svm_grid.fit(X_train_scaled, y_train)
    models['SVM'] = svm_grid.best_estimator_
    joblib.dump(svm_grid.best_estimator_, r'')
    cnn_model.load_state_dict(torch.load(r''))
    cnn_model.eval()
    y_pred_cnn = []
    with torch.no_grad():
        for val_features, val_labels in val_loader:
            outputs, _ = cnn_model(val_features)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_cnn.extend(predicted.numpy())
    models['CNN'] = {
        'predictions': y_pred_cnn,
        'true_labels': y_true_list
    }

    confusion_matrices = {}

    for name, model in models.items():
        if name == 'CNN':
            y_pred = model['predictions']
            y_true = model['true_labels']
        else:
            y_pred = model.predict(X_val_scaled)
            y_true = y_val

        cm = confusion_matrix(y_true, y_pred)
        confusion_matrices[name] = cm

        cm_df = pd.DataFrame(cm)
        with open(r'', 'a') as f:
            f.write(f"\n{name} Confusion Matrix:\n")
        cm_df.to_csv(r'', mode='a', header=False)

train_and_evaluate_models()