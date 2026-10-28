import torch

from data import load_data, split_data, scale_data, make_tensor_datasets
from model import CancerClassifier
from train import train_model

from torch.utils.data import DataLoader
from plotting import plot_learning_rate_comparison

def prepare_data(batch_size=32):
    X, y = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train, X_val, X_test, _ = scale_data(
        X_train, X_val, X_test
    )

    train_dataset, val_dataset, test_dataset = make_tensor_datasets(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader

def run_experiment(lr, initial_weights, train_loader, val_loader, epochs=100):
    model=CancerClassifier()
    model.load_state_dict(initial_weights)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_history, val_history, best_epoch, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        epochs,
    )

    return train_history, val_history, best_epoch, best_val_loss

# if __name__ == "__main__":
#     torch.manual_seed(42)

#     results = {}
#     batch_sizes = [16,32,64]
#     learning_rate = 0.03

#     for batch_size in batch_sizes:
#         print(f"\nBatch size: {batch_size}")

#         train_loader, val_loader = prepare_data(batch_size=batch_size)

#         # Create the same starting weights for every experiment.
#         torch.manual_seed(42)
#         base_model=CancerClassifier()
#         initial_weights = { 
#             key: value.clone() 
#             for key, value in base_model.state_dict().items() 
#         }

#         train_history, val_history, best_epoch, best_val_loss = run_experiment(
#             learning_rate,
#             initial_weights,
#             train_loader,
#             val_loader,
#             epochs=100
#         )

#         results[batch_size] = {
#             "train": train_history,
#             "validation": val_history,
#             "best_epoch": best_epoch,
#             "best_val_loss": best_val_loss,
#         }

#     print("\nFinal validation losses:")
#     for batch_size, history in results.items():
#         print(
#             f"Batch size {batch_size} | "
#             f"Final val loss: {history['validation'][-1]:.4f} | "
#             f"Best val loss: {history['best_val_loss']:.4f} | "
#             f"Best epoch: {history['best_epoch']}"
#         )
    