import logging
from abc import ABC, abstractmethod

import numpy as np

class Evaluation(ABC):
    """
    Abstract class defining strategy for model evaluation.

    """
    def calculate_scores(self , y_true: np.ndarray, y_pred: np.ndarray):
        """
        Abstract method to calculate evaluation scores.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Concrete class implementing Mean Squared Error (MSE) evaluation strategy.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """
        Calculate Mean Squared Error (MSE) between true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Mean Squared Error score.
        """
        try:
            mse = np.mean((y_true - y_pred) ** 2)
            logging.info(f"Mean Squared Error calculated: {mse}")
            return np.float64(mse)
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            raise e
        
class R2(Evaluation):
    """
    Concrete class implementing R-squared evaluation strategy.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """
        Calculate R-squared score between true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: R-squared score.
        """
        try:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
            logging.info(f"R-squared score calculated: {r2_score}")
            return np.float64(r2_score)
        except Exception as e:
            logging.error(f"Error calculating R-squared score: {e}")
            raise e 
        
class RMSE(Evaluation):
    """
    Concrete class implementing Root Mean Squared Error (RMSE) evaluation strategy.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        """
        Calculate Root Mean Squared Error (RMSE) between true and predicted labels.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Root Mean Squared Error score.
        """
        try:
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            logging.info(f"Root Mean Squared Error calculated: {rmse}")
            return np.float64(rmse)
        except Exception as e:
            logging.error(f"Error calculating RMSE: {e}")
            raise e 
        
        