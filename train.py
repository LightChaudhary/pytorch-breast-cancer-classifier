import torch
from torch.utils.data import DataLoader

from data import load_data, split_data, scale_data, make_tensor_datasets
from model import CancerClassifier
from plotting import plot_loss
from evaluate import evaluate_model

def prepare_data(batch_size=32):
    X, y = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X,y)

    X_train, X_val, X_test, _ = scale_data(
        X_train, X_val, X_test
    )

    train_dataset, val_dataset, test_dataset = make_tensor_datasets(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        # 1. Clear gradients from the previous batch
        optimizer.zero_grad()

        # 2. Forward pass
        logits = model(X_batch)

        # 3. Calculate loss
        loss = loss_fn(logits, y_batch)

        # 4. Backpropagation
        loss.backward()

        # 5. Update model parameters
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(train_loader.dataset)

def evaluate_loss(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

    average_loss = total_loss / len(data_loader.dataset)

    return average_loss

def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=100,):
    train_history = []
    val_history = []

    best_epoch = 0
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            loss_fn,
        )

        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = val_loss

            best_weights = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f}")

    model.load_state_dict(best_weights)

    return train_history, val_history, best_epoch, best_val_loss

# if __name__ == "__main__":
#     train_loader, val_loader, test_loader = prepare_data()

#     model = CancerClassifier()
#     loss_fn = torch.nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=0.003,
#     )

#     train_history, val_history, best_epoch, best_val_loss = train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         loss_fn=loss_fn,
#         optimizer=optimizer,
#     )

#     print("Final Train loss:", round(train_history[-1], 4))
#     print("Final Val loss:", round(val_history[-1], 4))

#     print("Best epoch:", best_epoch)
#     print("Best validation loss:", round(best_val_loss, 4))
#     evaluate_model(model, test_loader)

#     plot_loss(train_history=train_history, val_history=val_history)