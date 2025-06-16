import logging 
from zenml import step
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict , Union 

class DataStrategy:
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Abstract method to handle data.
        
        Args:
            data (pd.DataFrame): DataFrame containing the data to be handled.
        
        Returns:
            Union[pd.DataFrame, Dict[str, Any]]: Processed data or metadata.
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
                        'order_approval_status',
                        'order_approval_status_reason',
                        'order_approval_status_reason_code',
                        'order_purchase_timestamp',

                    ]
                    ,axis =1)
            
            data['product_weight_g'] = data['product_weight_g'].fillna(0)
            data['product_length_cm'] = data['product_length_cm'].fillna(0) 
            data['product_height_cm'] = data['product_height_cm'].fillna(0)
            data['product_width_cm'] = data['product_width_cm'].fillna(0)
            data['review_comment_message'] = data['review_comment_message'].fillna('')
            data['review_comment_title'] = data['review_comment_title'].fillna('')

            data = data.select_dtypes(include=['number'])
            col_to_drop = data.columns[data.isnull().mean() > 0.5]
            data = data.drop(columns=col_to_drop, axis=1) 
            return data
        except Exception as e:
            logging.error(f"Error in DataPreProcessingStrategy: {e}")
            raise e


