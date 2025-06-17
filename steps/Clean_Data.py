import logging 
import pandas as pd
from zenml import step
from typing import Any, Tuple
from typing_extensions import Annotated 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.Data_Cleaning import DataCleaning, DataPreProcessingStrategy, DataDivideStrategy, DataEncodingStrategy


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "x_train"], 
    Annotated[pd.DataFrame, "x_test"], 
    Annotated[pd.Series, "y_train"], 
    Annotated[pd.Series, "y_test"]]:

    """    Cleans and preprocesses the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned and preprocessed.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training and testing sets for features and labels.
    Raises: 
        Exception: If an error occurs during the data cleaning and preprocessing steps.
    Logs:   
        INFO: Data preprocessed successfully.
        INFO: Data encoded successfully.
        INFO: Data cleaned and divided successfully.
        ERROR: Error occurred in data cleaning step.
    Example:
        >>> import pandas as pd
        >>> from steps.Clean_Data import clean_data
        >>> df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'label': [0, 1, 0, 1]
        })
        >>> x_train, x_test, y_train, y_test = clean_data(df)
        >>> print(x_train)
        feature1  feature2
        0        1        5
        1        2        6

        >>> print(y_train)
        0    0
        1    1
        Name: label, dtype: int64

        >>> print(x_test)
        feature1  feature2
        2        3        7
        3        4        8

        >>> print(y_test)
        2    0
        3    1
        Name: label, dtype: int64

    """


    try:
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        logging.info("Data preprocessed successfully.")

        encoding_strategy = DataEncodingStrategy()
        data_cleaning = DataCleaning(processed_data, encoding_strategy) 
        processed_data = data_cleaning.handle_data()
        logging.info("Data encoded successfully.")

        dividestrategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, dividestrategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaned and divided successfully.")
        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        logging.error(f"Error occurred in data cleaning step: {e}")
        raise e 
    return x_train, x_test, y_train, y_test

