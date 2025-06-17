import logging
from abc import ABC, abstractmethod
from typing import Any, Tuple
from sklearn.linear_model import LinearRegression

class Model(ABC):
    @abstractmethod
    def train(self, X_train: Any, y_train: Any):
        """
        Abstract method to train the model with the provided data.
        
        Args:
            X_train (Any): Features to be used for training the model.
            y_train (Any): Labels to be used for training the model.
        """
        pass



class LinearRegressionModel(Model):
    def train(self, X_train: Any, y_train: Any, **kwargs):
        """
        Train the linear regression model with the provided data.
        
        Args:
            X_train (Any): Features to be used for training the model.
            y_train (Any): Labels  to be used for training the model.
            kwargs: Additional keyword arguments for training configuration.
        Returns:
            None

        """

        # Implement training logic here

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model trained successfully.")
            return reg
        
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise e
