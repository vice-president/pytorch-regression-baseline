import torch
from torch import nn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=2200, n_features=25, noise=12.0, random_state=11)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
y_val = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
y_val = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

model = nn.Sequential(nn.Linear(25, 128), nn.ReLU(), nn.Linear(128, 1))
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 61):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        mse = loss_fn(val_pred, y_val).item()
        mae = torch.mean(torch.abs(val_pred - y_val)).item()
    if epoch % 10 == 0:
        print(f"epoch={epoch:02d} train_mse={loss.item():.4f} val_mse={mse:.4f} val_mae={mae:.4f}")
