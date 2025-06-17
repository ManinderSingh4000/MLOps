import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
import sys
import os
# Add project root (one level above 'steps') to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.model_dev import LinearRegressionModel  # type: ignore # Updated import to absolute path

def model_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series

) :
    """
    Step to train a linear regression model on the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the features and labels for training.
    
    Returns:
        None
    """
    
    model = LinearRegressionModel()
    model.train(X_train, y_train,)
    return model # type: ignore
