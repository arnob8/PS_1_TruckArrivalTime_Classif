import sys # provides access to system specific parameter and funcs, used to interact with Python runtime env
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer #used to create the pipeline
from sklearn.impute import SimpleImputer # to impute missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object # jsut used for saving the pickle fil

import os

#We just want the input to the DataTransformationConfig
@dataclass
class DataTransformationConfig:
    train_data_transformed_path: str=os.path.join("artifacts","train_transformed.csv") #all data will be saved in artifacts path, filename train.csv
    #os.path dynamically adjust / or \ based on os
    #output => artifacts\train.csv
    test_data_transformed_path: str=os.path.join("artifacts","test_transformed.csv") #all data will be saved in artifacts path
    

class DataTransformation1:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def transformation_on_orginal_dataset(self,df):
        '''
        This function is responsible for data transformation, on the original dataset, to 
        create a intermediary dataset.
        Includes combining Planned date and time columns, label encoding, and adding derived columns.
        '''
        try:
            # Transformation 1:Combine date and time columns into datetime
            logging.info("Combining date and time columns into datetime.")
            df["PlannedDateTime"] = pd.to_datetime(
                df["Planned Date"] + " " + df["Planned Time"],
                format="%d/%m/%Y %I:%M:%S %p",
            )
            df["ArrivedDateTime"] = pd.to_datetime(
                df["Arrival Date"] + " " + df["Arrival Time"],
                format="%d/%m/%Y %I:%M:%S %p",
            )

            # Initialize LabelEncoder
            label_encoder = LabelEncoder()

            # Transformation 2:Perform label encoding on Transport Company,RelationCode,Customer
            logging.info("Encoding categorical columns with LabelEncoder.")
            df["CarrierID"] = label_encoder.fit_transform(df["Transport Company"])
            df["RelationID"] = label_encoder.fit_transform(df["RelationCode"])
            df["CustomerID"] = label_encoder.fit_transform(df["Customer"])

            # Transformation 3:Add derived column,NumberOfOrders against each TripId
            logging.info("Adding the 'NumberOfOrders' column.")
            df["NumberOfOrders"] = 1

            # Transformation 4:Select and return the required columns
            logging.info("Dropping unnecessary columns and selecting required columns.")
            df_transformed = df[
                [
                    "Date",
                    "CarrierID",
                    "RelationID",
                    "Trip Nr",
                    "Order type",
                    "CustomerID",
                    "PlannedDateTime",
                    "ArrivedDateTime",
                    "NumberOfOrders",
                ]
            ]

            logging.info("Data transformation completed successfully.")
            return df_transformed
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_firstlevel_data_transfor(self,train_path,test_path):

        try:
            # Step 1: Read train and test data
            logging.info("Reading train and test data from CSV files.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data successfully read.")

            # Step 2: Apply transformations
            logging.info("Applying transformations on train and test datasets.")
            train_transformed_df = self.transformation_on_orginal_dataset(train_df)
            test_transformed_df = self.transformation_on_orginal_dataset(test_df)

            # Step 3: Save the transformed data to CSV files
            logging.info("Saving transformed train and test datasets.")
            train_transformed_df.to_csv(
                self.data_transformation_config.train_data_transformed_path, index=False, header=True
            )
            test_transformed_df.to_csv(
                self.data_transformation_config.test_data_transformed_path, index=False, header=True
            )

            logging.info("Transformation of the datasets into new CSV files is completed.")

            return (
                # Step 4: Returning the path of the transformed csv file paths for next step
                self.data_transformation_config.train_data_transformed_path,
                self.data_transformation_config.test_data_transformed_path,
            )
        except Exception as e:
            raise CustomException(e,sys)