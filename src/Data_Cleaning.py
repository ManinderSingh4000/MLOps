import logging 
import pandas as pd
from abc import abstractmethod
from typing import Any, Tuple
from sklearn.preprocessing import LabelEncoder
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
                        'stories',
                        'mainroad',
                        'parking',
                        'prefarea'

                    ]
                    ,axis =1 , inplace=True)
            data = data.dropna()
            data = data.drop_duplicates()
            data = data.reset_index(drop=True)
            data = data.select_dtypes(include=['number'])
            col_to_drop = data.columns[data.isnull().mean() > 0.5]
            data = data.drop(columns=col_to_drop, axis=1) 
            return data
        except Exception as e:
            logging.error(f"Error in DataPreProcessingStrategy: {e}")
            raise e
        
class DataEncodingStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables in the data.
        
        Args:
            data (pd.DataFrame): DataFrame containing the data to be encoded.
        
        Returns:
            pd.DataFrame: Encoded DataFrame.
        """
        try:
            encoder = LabelEncoder()

            for i in data.select_dtypes(include=['object']).columns:
                data[i] = encoder.fit_transform(data[i])
            return data
        except Exception as e:
            logging.error(f"Error in DataEncodingStrategy: {e}")
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
            X = data.drop(["price"], axis=1)
            y = data[["price"]]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return x_train, x_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Error in DataDivideStrategy: {e}")
            raise e

class DataCleaning:
    def __init__(self , data : pd.DataFrame ,  strategy: DataStrategy):

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
        
# if __name__ == "__main__":
#     data = pd.read_csv("C:\\Users\\python\\OneDrive\\Desktop\\MLOps\\Data\\Housing.csv")
#     data_cleaning = DataCleaning(data, DataPreProcessingStrategy())
#     data_cleaning.handle_data()

#     print("\nData cleaned successfully.\n")
#     print(data_cleaning.data.head())

#     data_encoding = DataCleaning(data_cleaning.data, DataEncodingStrategy())
#     data_cleaning.data = data_encoding.handle_data()
#     print("\nData encoded successfully.")
#     print(data_cleaning.data.head())


#     data_divide = DataCleaning(data_cleaning.data, DataDivideStrategy())
#     x_train, x_test, y_train, y_test = data_divide.handle_data()

#     print("\nData divided successfully.")
#     print(f"x_train shape: {x_train.shape}")
#     print(f"x_test shape: {x_test.shape}")
#     print(f"y_train shape: {y_train.shape}")
#     print(f"y_test shape: {y_test.shape}")

