"""
Accuracy: how often predictions are correct
Precision: of the tumors predicted as malignant, how many actually are malignant?
Recall: of all actual malignant tumors, how many did the model catch?
F1-score: A balance between precision and recall
"""

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, test_loader):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            all_labels.extend(y_batch.squeeze(1).tolist())
            all_predictions.extend(predictions.squeeze(1).tolist())

    # Dataset label 0 = malignant, so treat 0 as positive.
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions, pos_label=0, zero_division=0
    )
    recall = recall_score(
        all_labels,all_predictions, pos_label=0, zero_division=0
    )
    f1 = f1_score(
        all_labels, all_predictions, pos_label=0, zero_division=0
    )

    matrix = confusion_matrix(all_labels, all_predictions, labels=[0,1])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion matrix(rows=actual, columns=predicted):")
    print(matrix)
