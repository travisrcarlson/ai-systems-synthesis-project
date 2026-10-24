"""The photo screener: a CNN in the layout of P4's baseline, trained on P4's CTX-UXO crops (P4).

It labels a field photo with one of P4's five categories and a confidence, or "unknown" when its confidence is
below a cut-off set on the validation crops, as P4 did: the lowest confidence at which the most confident
validation crops still reach 95% accuracy. Its reading is a first indication for an explosive ordnance disposal
technician to confirm. It is not evidence, and it decides nothing.
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CATEGORIES = ["Projectile", "Grenade", "Mortar Bomb", "Cartridge", "Aviation Bomb"]  # P4's five, largest first
TARGET_ACCURACY = 0.95
SETTINGS = {"epochs": 20, "batch": 64, "learning_rate": 0.001, "dropout": 0.3, "threads": 2}  # P4's settings


def conv_block(channels_in, channels_out):
    """3 x 3 convolution, batch normalisation, ReLU, then 2 x 2 max pooling."""
    return nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(channels_out), nn.ReLU(), nn.MaxPool2d(2))


class ScreenerCNN(nn.Module):
    """Four convolution blocks (32, 64, 128, 256 channels), global average pooling, dropout and a linear layer."""

    def __init__(self, n_classes=len(CATEGORIES), dropout=SETTINGS["dropout"]):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, 256))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def standardise(images, mean, std):
    """8-bit crops (n, 64, 64, 3) to a float tensor (n, 3, 64, 64), standardised with the training statistics."""
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()
    x = x.contiguous(memory_format=torch.channels_last).div_(255)
    return x.sub_(mean[:, None, None]).div_(std[:, None, None])


def flip_and_turn(batch, generator):
    """Flip each crop left to right with probability 0.5, then turn it by a random multiple of 90 degrees."""
    flips = torch.rand(len(batch), generator=generator) < 0.5
    batch = torch.where(flips[:, None, None, None], batch.flip(-1), batch)
    turns = torch.randint(0, 4, (len(batch),), generator=generator)
    out = batch.clone()
    for k in (1, 2, 3):
        chosen = turns == k
        if chosen.any():
            out[chosen] = torch.rot90(batch[chosen], k, dims=(-2, -1))
    return out


def logits(model, x, batch=512):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


def scores(y_true, y_pred, k=len(CATEGORIES)):
    """Accuracy and macro-F1 over k categories."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    matrix = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(k), where=matrix.sum(0) > 0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(k), where=matrix.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=precision + recall > 0)
    return {"accuracy": tp.sum() / matrix.sum(), "macro_f1": f1.mean()}


def train(train_images, train_labels, valid_images, valid_labels, settings=None, seed=0, verbose=True):
    """Train a fresh screener; keep the weights of the epoch with the best validation macro-F1.

    Returns the model in evaluation mode, its standardisation statistics, the training history and the best epoch.
    """
    s = {**SETTINGS, **(settings or {})}
    torch.set_num_threads(s["threads"])        # fixed, so results do not depend on the computer's core count
    torch.use_deterministic_algorithms(True)
    unit = torch.from_numpy(train_images).permute(0, 3, 1, 2).float().div(255)
    mean, std = unit.mean(dim=(0, 2, 3)), unit.std(dim=(0, 2, 3), correction=0)
    del unit
    x, y = standardise(train_images, mean, std), torch.from_numpy(np.asarray(train_labels))
    xv, yv = standardise(valid_images, mean, std), torch.from_numpy(np.asarray(valid_labels))
    counts = torch.bincount(y, minlength=len(CATEGORIES)).float()
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (len(CATEGORIES) * counts))
    torch.manual_seed(seed)
    model = ScreenerCNN().to(memory_format=torch.channels_last)
    optimiser = torch.optim.Adam(model.parameters(), lr=s["learning_rate"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=s["epochs"])
    order_gen, turn_gen = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 1)
    history, best, started = [], (-1.0, 0, None), time.perf_counter()
    for epoch in range(1, s["epochs"] + 1):
        model.train()
        order, running = torch.randperm(len(x), generator=order_gen), 0.0
        for start in range(0, len(x), s["batch"]):
            batch = order[start:start + s["batch"]]
            optimiser.zero_grad()
            loss = loss_fn(model(flip_and_turn(x[batch], turn_gen)), y[batch])
            loss.backward()
            optimiser.step()
            running += loss.item() * len(batch)
        schedule.step()
        out = logits(model, xv)
        valid = scores(yv.numpy(), out.argmax(1).numpy())
        history.append({"epoch": epoch, "train loss": running / len(x), "validation loss": loss_fn(out, yv).item(),
                        "validation accuracy": valid["accuracy"], "validation macro-F1": valid["macro_f1"]})
        if valid["macro_f1"] > best[0]:
            best = (valid["macro_f1"], epoch, copy.deepcopy(model.state_dict()))
        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(f"  epoch {epoch:>2}: validation macro-F1 {valid['macro_f1']:.3f}, "
                  f"{time.perf_counter() - started:.0f} s", flush=True)
    model.load_state_dict(best[2])
    model.eval()
    return model, mean, std, pd.DataFrame(history).set_index("epoch"), best[1]


def probabilities(model, images, mean, std):
    return logits(model, standardise(images, mean, std)).softmax(1)


def choose_cutoff(probs, y, target=TARGET_ACCURACY):
    """The lowest confidence at which the most confident crops still reach the target accuracy (P4's rule)."""
    confidence, predicted = probs.max(1)
    order = torch.argsort(confidence, descending=True, stable=True)
    correct = (predicted[order] == torch.as_tensor(y)[order]).double()
    accuracy = (correct.cumsum(0) / torch.arange(1, len(order) + 1)).numpy()
    reached = np.flatnonzero(accuracy >= target)
    return float(confidence[order][reached[-1]]) if len(reached) else 1.0


def readings(probs, cutoff):
    """Each crop's reading: (category or "unknown", confidence)."""
    confidence, predicted = probs.max(1)
    return [(CATEGORIES[int(p)] if c >= cutoff else "unknown", float(c)) for c, p in zip(confidence, predicted)]


def reading_text(reading, cutoff):
    label, confidence = reading
    if label == "unknown":
        return f"unknown (confidence {confidence:.2f}, below the cut-off of {cutoff:.2f}); for a technician to identify"
    return f"possibly {label.lower()} (confidence {confidence:.2f}); for a technician to confirm"


def save(path, model, mean, std, cutoff, history, best_epoch):
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "cutoff": cutoff,
                "history": history.reset_index().to_dict("list"), "best_epoch": best_epoch,
                "categories": CATEGORIES, "settings": SETTINGS}, path)


def load(path):
    saved = torch.load(path, weights_only=False)
    model = ScreenerCNN().to(memory_format=torch.channels_last)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    history = pd.DataFrame(saved["history"]).set_index("epoch")
    return model, saved["mean"], saved["std"], saved["cutoff"], history, saved["best_epoch"]
