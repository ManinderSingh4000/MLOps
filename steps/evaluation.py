import logging
from typing import Tuple , Annotated
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from src.evaluation import MSE , R2 , RMSE

@step
def evaluate_model(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model,  # Assuming model is a scikit-learn compatible
) -> Tuple[
    Annotated['float','r2_score'],
    Annotated['float','rmse_score']
]:
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test.values, prediction)
        logging.info(f"Mean Squared Error: {mse}")

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test.values, prediction)
        logging.info(f"R2 Score: {r2}")     

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test.values, prediction)
        logging.info(f"Root Mean Squared Error: {rmse}")

        return r2, rmse
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise e