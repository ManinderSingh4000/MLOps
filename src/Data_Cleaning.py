import logging 
from zenml import step
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Tuple
from sklearn.model_selection import train_test_split

class DataStrategy:
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Any:
        """
        Abstract method to handle data.
        
        Args:
            data (pd.DataFrame): DataFrame containing the data to be handled.
        
        Returns:
            Any: Processed data or metadata.
        """
        pass

class DataPreProcessingStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by removing NaN values and duplicates.
        
        Args:
            data (pd.DataFrame): DataFrame containing the data to be preprocessed.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        try:
            data.drop(columns=
                    [
                        'review_comment_message',

                    ]
                    ,axis =1)
            
            data['product_weight_g'] = data['product_weight_g'].fillna(0)
            data['product_length_cm'] = data['product_length_cm'].fillna(0) 
            data['product_height_cm'] = data['product_height_cm'].fillna(0)
            data['product_width_cm'] = data['product_width_cm'].fillna(0)
            data['review_comment_message'] = data['review_comment_message'].fillna('')
            # data['review_comment_title'] = data['review_comment_title'].fillna('')

            data = data.select_dtypes(include=['number'])
            col_to_drop = data.columns[data.isnull().mean() > 0.5]
            data = data.drop(columns=col_to_drop, axis=1) 
            return data
        except Exception as e:
            logging.error(f"Error in DataPreProcessingStrategy: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide the data into training and testing sets.
        
        Args:
            data (pd.DataFrame): DataFrame containing the data to be divided.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training and testing sets for features and target.
        """
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return x_train, x_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Error in DataDivideStrategy: {e}")
            raise e

class DataCleaning:
    def __init__(self , data ,  strategy: DataStrategy):

        """
        Initialize the DataCleaning class with a specific strategy.
        
        Args:
            strategy (DataStrategy): Strategy to be used for data handling.
        """
        self.strategy = strategy
        self.data = data

    def handle_data(self) -> Any:
        """
        Handle the data using the specified strategy.
        """

        try:
            return self.strategy.handle_data(self.data)
        
        except Exception as e:


            logging.error(f"Error in DataCleaning: {e}")
            raise e
        
if __name__ == "__main__":
    data = pd.read_csv("C:\\Users\\python\\OneDrive\\Desktop\\MLOps\\Data\\olist_customers_dataset.csv")
    data_cleaning = DataCleaning(data, DataPreProcessingStrategy())
    data_cleaning.handle_data()

