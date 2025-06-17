import logging
from zenml import  pipeline
from steps.Data_Ingestion import data_ingestion
from steps.Clean_Data import clean_data
from steps.evaluation import evaluate_model
from steps.Model_Train import model_train

@pipeline
def train_pipeline(data_path: str) :
  
    try:
        # Step 1: Data Ingestion
        df = data_ingestion(data_path=data_path)

        # Step 2: Data Cleaning
        x_train, x_test, y_train, y_test = clean_data(df=df)
        logging.info("Data cleaning completed successfully.")

        # Step 3: Model Training
        # Provide the required config argument to model_train
        # If ModelNameConfig does not exist, remove its usage and pass required arguments directly or set config to None
          # or provide appropriate config if needed
        model = model_train(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
        # print(model)
         # Return the trained model if needed
        # logging.info("Model training completed successfully.")

        # r2_score, rmse_score = evaluate_model(X_test=x_test, y_test=y_test, model=model)
        # logging.info(f"Model evaluation completed with R2 score: {r2_score} and RMSE score: {rmse_score}")

        

    except Exception as e:
        logging.error(f"Error occurred in the training pipeline: {e}")