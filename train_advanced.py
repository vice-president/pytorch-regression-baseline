import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def make_data(n_samples=4000, n_features=30, noise=15.0):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=SEED)
    return X, y


class Regressor(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_one_split(X_train, y_train, X_val, y_val, epochs=200, batch_size=128):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32))
    val_x = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = Regressor(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    best = {"val_loss": float("inf"), "state": None, "epoch": 0}
    patience, patience_left = 20, 20

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = loss_fn(val_pred, val_y).item()

        scheduler.step(val_loss)

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "state": model.state_dict(), "epoch": epoch}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pred = model(val_x).numpy().reshape(-1)

    rmse = mean_squared_error(y_val, pred, squared=False)
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)

    return {
        "best_epoch": best["epoch"],
        "best_val_mse": best["val_loss"],
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def run_kfold_cv(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_metrics = []

    for i, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[tr_idx], X[va_idx]
        y_train, y_val = y[tr_idx], y[va_idx]

        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.transform(X_val)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)

        result = train_one_split(X_train, y_train, X_val, y_val_scaled)
        fold_metrics.append(result)
        print(f"fold={i} rmse={result['rmse']:.4f} mae={result['mae']:.4f} r2={result['r2']:.4f} best_epoch={result['best_epoch']}")

    summary = {
        "k_folds": k,
        "avg_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "avg_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "avg_r2": float(np.mean([m["r2"] for m in fold_metrics])),
        "fold_metrics": fold_metrics,
    }
    return summary


if __name__ == "__main__":
    X, y = make_data()

    # quick holdout run
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    xs, ys = StandardScaler(), StandardScaler()
    X_train = xs.fit_transform(X_train)
    X_val = xs.transform(X_val)
    y_train = ys.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val_scaled = ys.transform(y_val.reshape(-1, 1)).reshape(-1)

    holdout = train_one_split(X_train, y_train, X_val, y_val_scaled)
    cv = run_kfold_cv(X, y, k=5)

    out = {"holdout": holdout, "cross_validation": cv}
    print(json.dumps(out, indent=2))
    with open("advanced_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
