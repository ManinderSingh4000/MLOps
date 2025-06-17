import logging
from zenml import  pipeline
from steps.Data_Ingestion import data_ingestion
from steps.Clean_Data import clean_data
from steps.Model_Train import model_train
from steps.Predictionn import make_prediction
# from steps.predict import make_prediction

from zenml.steps import step



@pipeline
def train_pipeline(data_path: str):
    try:
        # Step 1: Data Ingestion
        df = data_ingestion(data_path=data_path)

        # Step 2: Data Cleaning
        x_train, x_test, y_train, y_test = clean_data(df=df)
        logging.info("✅ Data cleaning completed successfully.")

        # Step 3: Model Training
        model = model_train(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
        logging.info("✅ Model training completed successfully.")

        # Step 4: Model Prediction
        y_pred = make_prediction(model=model, X=x_test)
        logging.info("✅ Model prediction completed successfully.")

        # # Step 5: Evaluation
        # evaluate_model(y_test=y_test, y_pred=y_pred)
        # logging.info("✅ Model evaluation completed successfully.")

    except Exception as e:
        logging.error(f"❌ Error occurred in the training pipeline: {e}")

