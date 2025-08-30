import os
import random
import numpy as np
import torch

""" Set random seed for reproducibility. """
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

""" Safely move tensor or numpy array to CPU and convert to numpy array. """
def safecpu(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

""" Returns explained variance between predictions and true values. """
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    return float(np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y)