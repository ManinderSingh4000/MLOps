import logging
from typing import Tuple , Annotated
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from src.evaluation import MSE , R2 , RMSE

@step
def evaluate_model(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model,  # Assuming model is a scikit-learn compatible
) -> Tuple[
   Annotated['float','r2_score'],
   Annotated['float','rmse_score']
]:

    try:
        prediction = model.predict(X_test)

        mse = MSE()
        r2 = R2()
        rmse = RMSE()

        import numpy as np
        y_true = np.asarray(y_test.values)
        y_pred = np.asarray(prediction)

        mse_score = mse.calculate_scores(y_true, y_pred)
        r2_score = r2.calculate_scores(y_true, y_pred)
        rmse_score = rmse.calculate_scores(y_true, y_pred)

        logging.info(f"Evaluation results - MSE: {mse_score}, R2: {r2_score}, RMSE: {rmse_score}")

        return r2_score, rmse_score
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise e 