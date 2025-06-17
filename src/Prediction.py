import logging

from zenml import step
from abc import ABC, abstractmethod

class PredictionStrategy(ABC):
    @abstractmethod
    def predict(self, model, X):
        """
        Abstract method to make predictions using the model.
        
        Args:
            model: The trained model to use for predictions.
            X (pd.DataFrame): Input features for prediction.
        
        Returns:
            pd.Series: Predicted values.
        """
        pass

class Prediction(PredictionStrategy):
    def predict(self, model, X):
        """
        Predicts the target variable using the trained model.

        Args:
            model: The trained model to use for predictions.
            X (pd.DataFrame): Input features for prediction.

        Returns:
            pd.Series: Predicted values.
        """
        return model.predict(X)

