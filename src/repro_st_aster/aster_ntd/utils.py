import numpy as np
import torch


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(MSE(y_true, y_pred)))


def MAPE(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


def R2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def generate_graph_Laplacian(A: np.ndarray) -> torch.Tensor:
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.diag(d) - A
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    return torch.from_numpy(L_norm.astype(np.float32))

