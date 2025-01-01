#Read the data and split the data into train and test
import os 
import sys # as we will use customer exception
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#Will uncomment later - Arnob - 01-01-2025
#from src.components.data_transformation import DataTransformation
#from src.components.data_transformation import DataTransformationConfig   
#dataclass is a decorator, if you are only defining variables then you can use data class, but if you have other 
#function, need init

#Will uncomment later - Arnob - 01-01-2025
#from src.components.model_trainer import ModelTrainer
#from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv") #all data will be saved in artifacts path, filename train.csv
    #os.path dynamically adjust / or \ based on os
    #output => artifacts\train.csv
    test_data_path: str=os.path.join("artifacts","test.csv") #all data will be saved in artifacts path
    raw_data_path: str=os.path.join("artifacts","data.csv") #all data will be saved in artifacts path

    #we can directly define the class variable without using init

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() 
        # When we call the class DataIngestionConfig the above 3 path will be saved in the path variables, they will
        #basically have the sub objects

    #Function 1
    def initiate_data_ingestion(self):
        #mongo db client can be present in utils
        logging.info("Entered the data ingestion method or component")
        try:
            #Step - 1 = Reading the dataset
            df = pd.read_csv(r"artifacts\Mel01ExportData_new.csv") #Here we can read from anywhere
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True) #Getting the directory name and not deleting
            # if its existing

            #Step -2 = Converted the Raw data path into csv file
            df.to_csv(self.ingestion_config.raw_data_path,index = False,header = True)

            logging.info("Train test split initiated")

            #Step -3 = Splitting the Train and Test data
            train_set,test_set = train_test_split(df,test_size = 0.2,random_state = 42)

            #Step -4 = Saving the train and test data
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)

            logging.info("Ingestion of the data is completed")

            return(
                #Step -5 = We pass the train data and test data path to the next step i.e Data Transformation
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()


#Will uncomment later - Arnob - 01-01-2025
    #data_transformation = DataTransformation() #It will call this -> self.data_transformation_config

    #Adding new steps to integrate Model Trainer
    #train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    #modeltrainer = ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_arr,test_arr)) #will return r2square