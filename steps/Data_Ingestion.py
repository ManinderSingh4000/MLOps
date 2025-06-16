
import logging
from zenml import step
import pandas as pd

class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) :

        logging.info(f"Loading data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def data_ingestion(data_path: str) -> pd.DataFrame:
    """
    Step to ingest data from a CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the ingested data.
    """
    try:
        ingestion = DataIngestion(data_path)
        df = ingestion.load_data()
        return df 
    except Exception as e:
       logging.error(f"Error occurred while ingesting data: {e}")
       return pd.DataFrame()  # Return an empty DataFrame on error