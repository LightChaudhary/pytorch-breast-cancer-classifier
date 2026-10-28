"""
Load the Breast Cancer dataset
Split it into training, validation and test sets
Scale the features without leaking information from validation or test data
Convert the prepared data into PyTorch tensors and datasets
"""

import numpy as np
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset

def load_data():
    dataset = load_breast_cancer()

    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.float32)

    return X,y

def split_data(X,y,seed=42):
    # Reserve 20% for testing
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X,y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    # take 20% of the temp datasets for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp,y_train_temp,
        test_size=0.20,
        random_state=42,
        stratify=y_train_temp,
    )

    return X_train,X_val,X_test,y_train,y_val,y_test

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()

    # Learn the mean and standard deviation from training data only
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply those same values to validation and test data
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return(
        X_train_scaled.astype(np.float32),
        X_val_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        scaler,
    )

def make_tensor_datasets(
  X_train, X_val, X_test,
  y_train, y_val, y_test,      
):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1,1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    return train_dataset, val_dataset, test_dataset

# if __name__ == "__main__":
#     X,y = load_data()

#     X_train,X_val,X_test,y_train,y_val,y_test = split_data(X,y)

#     X_train,X_val,X_test,scaler = scale_data(X_train,X_val,X_test)

#     print("Train:", X_train.shape, y_train.shape)
#     print("Validation:", X_val.shape, y_val.shape)
#     print("Test:", X_test.shape, y_test.shape)

#     print("First 5 scaled features:", X_train[0,:5])
#     print("Training feature means:", X_train.mean(axis=0)[:5])

#     train_ds, val_ds, test_ds = make_tensor_datasets(
#         X_train,X_val,X_test,
#         y_train,y_val,y_test,
#     )

#     features, labels = next(iter(train_ds))

#     print("One sample's feature shape:", features.shape)
#     print("One sample's label shape:", labels.shape)
#     print("Training dataset size :", len(train_ds))