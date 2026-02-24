# Advanced Regression Concepts (PyTorch)

This file explains the advanced ideas included in `train_advanced.py`.

## What is included

1. **AdamW + weight decay**
   - Better regularization than plain Adam for many tabular tasks.
2. **Dropout in hidden layers**
   - Reduces overfitting by adding stochastic regularization.
3. **Learning-rate scheduling** (`ReduceLROnPlateau`)
   - Automatically lowers LR when validation loss stalls.
4. **Early stopping**
   - Stops training when there is no validation improvement.
5. **Gradient clipping**
   - Improves stability by preventing exploding gradients.
6. **K-Fold Cross Validation (5-fold)**
   - Gives a more robust estimate of model quality than single split.

## Metrics reported
- RMSE
- MAE
- R²
- Best validation epoch

## Usage
```bash
python train_advanced.py
```
This writes `advanced_metrics.json` with holdout + CV metrics.
