import logging 

from Pipelines.Train_Pipeline import train_pipeline

if __name__ == "__main__":

    data_path = "C:\\Users\\python\\OneDrive\\Desktop\\MLOps\\Data\\Housing.csv"  # Path to your data file
    train_pipeline(data_path=data_path)