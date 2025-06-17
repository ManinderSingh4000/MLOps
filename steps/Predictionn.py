from src.Prediction import Prediction
from zenml import step
from typing import Annotated
import pandas as pd
import logging

# @step
# def make_prediction(
#     model,
#     X: Annotated[pd.DataFrame, "Input features for prediction"]
# ) -> Annotated[pd.Series, "Predicted values"]:
#     """
#     Step to make predictions using the trained model.

#     Args:
#         model (object): The trained model to use for predictions.
#         X (pd.DataFrame): Input features for prediction.

#     Returns:
#         pd.Series: Predicted values.
#     """
#     try:
#         logging.info("Starting prediction...")
#         predictor = Prediction()
#         predictions = predictor.predict(model, X)
#         logging.info("Prediction completed successfully.")
#         return predictions
#     except Exception as e:
#         logging.error(f"Error during prediction: {e}")
#         raise e


@step
def make_prediction(
    model,
    X: Annotated[pd.DataFrame, "Input features for prediction"]
) -> Annotated[pd.Series, "Predicted values"]:
    """
    Step to make predictions using the trained model.
    """
    try:
        import pandas as pd
        import numpy as np
        logging.info("Starting prediction...")
        predictor = Prediction()
        predictions = predictor.predict(model, X)

        # ✅ Convert np.ndarray to pd.Series before returning
        predictions_series = pd.Series(predictions, index=X.index)  # Optional: preserve index

        logging.info("Prediction completed successfully.")
        return predictions_series

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise e
