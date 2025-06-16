import logging
from zenml import step , pipeline
from steps.Data_Ingestion import data_ingestion
from steps.Clean_Data import clean_data
from steps.Model_Train import model_train

@pipeline
def train_pipeline(data_path: str) -> None:
    """
    Pipeline to ingest, clean, and train a model on the data.

    Args:
        data_path (str): Path to the CSV file containing the data.
        model: The model to be trained.
    """
    try:
        # Step 1: Data Ingestion
        df = data_ingestion(data_path=data_path)

        # Step 2: Data Cleaning
        cleaned_df = clean_data(df=df)

        # Step 3: Model Training
        model_train(df)
        
    except Exception as e:
        logging.error(f"Error occurred in the training pipeline: {e}")