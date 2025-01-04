import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object # to load our pickle file
from datetime import datetime,date,time

#First Class -> has the init function without nothing, 
class PredictPipeline:
    def __init__(self):
        pass

    #will simply do prediction
    # two pckle files we have currently, preprocessor and  model
    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            #load_obect we will craete, will load the pickle file
            model = load_object(file_path = model_path) #should be created in utils
            preprocessor = load_object(file_path = preprocessor_path)
            #print("Exploring the preprocessor object")
            #print(type(preprocessor))
            #print(preprocessor.transformers)
            data_scaled = preprocessor.transform(features)
            print("Exploring the model object")
            print(type(model))
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)
        


#Second Class -> Responsible for matching all the input we are passing in the html to the backend
class CustomData:
    def __init__( self,           
            date: date,
            transport_company: str,
            relation_name: str,
            relation_code: str,
            trip_nr: str,
            order_number: str,
            external_reference: str,
            order_type: str,
            customer: str,
            planned_date: date,
            planned_time: time,
            arrival_date: date,
            arrival_time: time):

            #Creating variable using self, the values are coming from web app agianst the respective variable
            self.date = date
            self.transport_company = transport_company
            self.relation_name = relation_name
            self.relation_code = relation_code
            self.trip_nr = trip_nr
            self.order_number = order_number
            self.external_reference = external_reference
            self.order_type = order_type
            self.customer = customer
            self.planned_date = planned_date
            self.planned_time = planned_time
            self.arrival_date = arrival_date
            self.arrival_time = arrival_time

    #It will basically return all our input in the form of a dataframe
    #From my web appplication , will get mapped to a datafram
    #could have been done in app.py but due to modularisation it is showed here 
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "date": [self.date],
                "transport_company": [self.transport_company],
                "relation_name":[self.relation_name],
                "relation_code":[self.relation_code],
                "trip_nr":[self.trip_nr],
                "order_number":[self.order_number],
                "external_reference":[self.external_reference],
                "order_type":[self.order_type],
                "customer":[self.customer],
                "planned_date":[self.planned_date],
                "planned_time":[self.planned_time],
                "arrival_date":[self.arrival_date],
                "arrival_time":[self.arrival_time],
                }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)