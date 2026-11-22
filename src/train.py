import copy
import time

import torch
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training data. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # clear old gradients
        outputs = model(images)            # forward pass: get predictions
        loss = criterion(outputs, labels)  # how wrong are they?
        loss.backward()                    # compute gradients
        optimizer.step()                   # update the weights

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate without learning. Returns (loss, accuracy, macro_f1, preds, labels)."""
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        n += images.size(0)
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / n, acc, macro_f1, all_preds, all_labels


def fit(model, train_loader, val_loader, criterion, optimizer, device,
        epochs=20, scheduler=None, save_path=None):
    """Full training loop. Keeps the epoch with the best validation macro F1."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        for key, value in zip(history, [tr_loss, tr_acc, val_loss, val_acc, val_f1]):
            history[key].append(value)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                torch.save(best_state, save_path)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} f1 {val_f1:.3f} | "
              f"{time.time() - start:.0f}s")

    model.load_state_dict(best_state)
    print(f"Best val macro F1: {best_f1:.3f}")
    return history