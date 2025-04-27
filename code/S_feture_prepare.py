import numpy as np
import pandas as pd
import torch
import joblib
'First, Load the trained base model ()'
'use joblib to load the parameters for base model'
    f1_dict_path = r''
    f1_dict_df = pd.read_csv(f1_dict_path)
    mlp_proba = mlp_model.predict_proba(X_train_scaled)
    rf_proba = rf_model.predict_proba(X_train_scaled)
    svm_proba = svm_model.predict_proba(X_train_scaled)
    X_train_tensor = torch.FloatTensor(np.expand_dims(X_train_scaled, axis=1))
    with torch.no_grad():
        cnn_outputs, _, cnn_attended_features = cnn_model(X_train_tensor)
        cnn_proba = torch.softmax(cnn_outputs, dim=1).numpy()
    cnn_pred_labels = torch.argmax(cnn_outputs, dim=1).numpy()
    weighted_probas = []
    for i, pred_label in enumerate(cnn_pred_labels):
        f1_row = f1_dict_df.iloc[pred_label].values[1:]
        model_probas = [
            mlp_proba[i],
            rf_proba[i],
            svm_proba[i],
            cnn_proba[i]
        ]
        weighted_proba = np.zeros_like(model_probas[0])
        for j, (proba, f1) in enumerate(zip(model_probas, f1_row)):
            weighted_proba += proba * f1

        weighted_probas.append(weighted_proba)

    weighted_probas = np.array(weighted_probas)

    output_df = pd.DataFrame(weighted_probas, columns=[f'Proba_{cls}' for cls in label_encoder.classes_])
    output_df.insert(0, 'Target', train_df['Target'])

    attended_features_df = pd.DataFrame(cnn_attended_features.numpy())
    attended_features_df.insert(0, 'Target', train_df['Target'])

    final_df = pd.concat([output_df, attended_features_df.iloc[:, 1:].add_prefix('AttendedFeature_')], axis=1)

    output_path = r''
    final_df.to_csv(output_path, index=False)

process_models_and_probabilities()