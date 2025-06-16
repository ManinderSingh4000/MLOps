import logging 
import pandas as pd
from zenml import step

@step
def clean_data(df: pd.DataFrame) -> None:
    pass
    # """
    # Step to clean the ingested data.

    # Args:
    #     df (pd.DataFrame): DataFrame containing the ingested data.

    # Returns:
    #     pd.DataFrame: Cleaned DataFrame.
    # """
    # try:
    #     logging.info("Starting data cleaning process")
        
    #     # Example cleaning operations
    #     df.dropna(inplace=True)  # Remove rows with missing values
    #     df.drop_duplicates(inplace=True)  # Remove duplicate rows
        
    #     logging.info("Data cleaning process completed successfully")
    #     return df
    # except Exception as e:
    #     logging.error(f"Error occurred while cleaning data: {e}")
    #     return pd.DataFrame()