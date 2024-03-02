import numpy as np
from pandas import Series


def calc_mse(pred: list | np.ndarray | Series, prob_occ: float = 1.0) -> float:
    """Calculate the mean square error of the predicted probability as treating the problem as regression

    Args:
        pred (list | np.ndarray | Series): predicted score
        prob_occ (float, optional): target probability of occurrence. Defaults to 1.0.

    Returns:
        float: result
    """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred, dtype=np.float32)

    return np.mean(np.square(pred - prob_occ)).item()


def calc_rmse(pred: list | np.ndarray | Series, prob_occ: float = 1.0) -> float:
    """Calculate root mean square error of the predicted probability as treating the problem as regression

    Args:
        pred (list | np.ndarray | Series): predicted scores
        prob_occ (float, optional): target probability of occurrence. Defaults to 1.0.

    Returns:
        float: result
    """
    return np.sqrt(calc_mse(pred, prob_occ))


def calc_tpr(pred: list | np.ndarray | Series, thres_occ: float = 0.5) -> float:
    """Calculate the True Positive Rate as treating the problem as one-class classification

    Args:
        pred (list | np.ndarray | Series): predicted probabbility
        thres_occ (float, optional): probability threshold of occurence. Defaults to 0.5.

    Returns:
        float: result
    """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred, dtype=np.float32)

    return (np.sum(pred >= thres_occ) * 1.0 / len(pred)).item()
