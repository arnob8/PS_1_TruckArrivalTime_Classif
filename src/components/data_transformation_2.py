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
class DataTransformationConfig2:
    train_data_final_path: str=os.path.join("artifacts","train_final.csv") #all data will be saved in artifacts path, filename train.csv
    #os.path dynamically adjust / or \ based on os
    #output => artifacts\train.csv
    test_data_final_path: str=os.path.join("artifacts","test_final.csv") #all data will be saved in artifacts path
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl") #Create models and want to save in a pkl file

class DataTransformation2:
    def __init__(self):
        self.data_transformation_config2 = DataTransformationConfig2()

###############STARTING WITH PREPROCESSING OF THE DATASET##############################33
    def get_data_transformer_object(self, df: pd.DataFrame) -> pd.DataFrame:
################################################
#3.4 - Deriving the Target Variable
        try:
            # TimeDifference = ArrivedTime - PlannedTime in minutes
            print("Shape Before Transformation - 3.4 - Derive Target Variable for Train:",df.shape)
            df['Delay'] = (df['ArrivedDateTime'] - df['PlannedDateTime']).dt.total_seconds() / 60  # in minutes
            df["TargetVariable"] = np.where(df["Delay"] >= 15, 1, 0)  # Target variable derived
            logging.info("Target variable derived successfully.")
            print("Shape After Transformation - 3.4 - Derive Target Variable for Train:",df.shape)
               
################################################
#3.5 - Removing Records with PlannedDateTime as NULL
    
            print("Shape Before Transformation - 3.5 - Remove Null Planned Time:",df.shape)
            df = df.dropna(subset=["PlannedDateTime"])
            logging.info(f"Records with PlannedDateTime Removed")
            print("Shape After Transformation - 3.5 - Remove Null Planned Time:",df.shape)
           
################################################
#3.6 - Removing Records with Duplicate Trip Nos
            print("Shape Before Transformation - 3.6 - Remove Duplicate Trip Nos:",df.shape)
            trip_count_multiple = df.groupby("Trip Nr")["Trip Nr"].count()
            trips_with_mult_counts = trip_count_multiple[trip_count_multiple > 1].index
            df_filtered = df[df["Trip Nr"].isin(trips_with_mult_counts)]
            df = df.drop(df_filtered.index)  # Removing duplicates
            logging.info("Duplicate trips removed successfully.")
            print("Shape After Transformation - 3.6 - Remove Duplicate Trip Nos:",df.shape)
            
################################################
#3.7 - Deriving TimeBasedFeatures from PlannedDateTime
            print("Shape Before Transformation - 3.7 - Deriving TimeBasedFeatures from PlannedDateTime:",df.shape)
            df['Planned_Hour'] = df['PlannedDateTime'].dt.hour
            df['Planned_Day'] = df['PlannedDateTime'].dt.day
            df['Planned_Weekday'] = df['PlannedDateTime'].dt.weekday  # Monday=0, Sunday=6
            df['Planned_Month'] = df['PlannedDateTime'].dt.month
            df['Planned_Year'] = df['PlannedDateTime'].dt.year
            df['Planned_Week'] = df['PlannedDateTime'].dt.isocalendar().week  # Weeks 1-52 in a year
        # Add IsWeekend column: 1 if weekend (Saturday=5, Sunday=6), 0 otherwise
            df['IsWeekend'] = df['Planned_Weekday'].apply(lambda x: 1 if x >= 5 else 0)
            logging.info("3.7 TimeBasedFeatures created successfully.")
            print("Shape After Transformation - 3.7 - Deriving TimeBasedFeatures from PlannedDateTime:",df.shape)
            
################################################
#3.8 - Deriving Cyclical & Frequency Features from columns created in 3.7
            print("Shape Before Transformation - 3.8 - Deriving Cyclical & Frequency Features from columns created in 3.7:",df.shape)
            ##########
            #1PlannedHour - Cyclic and Frequency Encoding
            ##########
            # Hourly Cyclic Encoding (0-23)
            df['Planned_Hour_sin'] = np.sin(2 * np.pi * df['Planned_Hour'] / 24)
            df['Planned_Hour_cos'] = np.cos(2 * np.pi * df['Planned_Hour'] / 24)

            def time_of_day(hour):
                if 6 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 18:
                    return 'Afternoon'
                elif 18 <= hour < 24:
                    return 'Evening'
                else:
                    return 'Night'

            df['Planned_TimeOfDay'] = df['Planned_Hour'].apply(time_of_day)

            # Step 2: Apply one-hot encoding to Planned_TimeOfDay
            time_of_day_encoded = pd.get_dummies(df['Planned_TimeOfDay'], prefix='Planned_TimeOfDay', drop_first=True).astype(int)  # Convert to integer type explicitly      
            # Step 3: Concatenate the original dataframe with the one-hot encoded columns
            df = pd.concat([df, time_of_day_encoded], axis=1)

            # Frequency Encoding #Based on the No Of Trips
            freq_map_plannedhour = df['Planned_Hour'].value_counts(normalize=True).to_dict()
            df['Planned_Hour_freq'] = df['Planned_Hour'].map(freq_map_plannedhour)

            ##########
            #2PlannedDay - Cyclic and Frequency Encoding
            ##########
            #Cyclic
            df['Planned_Day_sin'] = np.sin(2 * np.pi * df['Planned_Day'] / 29)
            df['Planned_Day_cos'] = np.cos(2 * np.pi * df['Planned_Day'] / 29)
            # Frequency Encoding #Based on the No Of Trips
            freq_map_plannedday = df['Planned_Day'].value_counts(normalize=True).to_dict()
            df['Planned_Day_freq'] = df['Planned_Day'].map(freq_map_plannedday)

            ##########
            #3PlannedWeekday - Cyclic and Frequency Encoding
            ##########
            #Cyclical Encoding
            df['Planned_Weekday_sin'] = np.sin(2 * np.pi * df['Planned_Weekday'] / 7)
            df['Planned_Weekday_cos'] = np.cos(2 * np.pi * df['Planned_Weekday'] / 7)


            # Frequency Encoding #Based on the No Of Trips
            freq_map = df['Planned_Weekday'].value_counts(normalize=True).to_dict()
            df['Planned_Weekday_freq'] = df['Planned_Weekday'].map(freq_map)

            ##########
            #4PlannedMonth - Cyclic and Frequency Encoding
            ##########
            #Cyclical Encoding
            df['Planned_Month_sin'] = np.sin(2 * np.pi * df['Planned_Month'] / 12)
            df['Planned_Month_cos'] = np.cos(2 * np.pi * df['Planned_Month'] / 12)


            # Frequency Encoding #Based on the No Of Trips
            freq_map_plannedmonth = df['Planned_Month'].value_counts(normalize=True).to_dict()
            df['Planned_Month_freq'] = df['Planned_Month'].map(freq_map_plannedmonth)

            ##########
            #5PlannedYear - Cyclic and Frequency Encoding
            ##########
            df['Planned_Year_sin'] = np.sin(2 * np.pi * df['Planned_Year'] / 12)
            df['Planned_Year_cos'] = np.cos(2 * np.pi * df['Planned_Year'] / 12)


            # Frequency Encoding #Based on the No Of Trips
            freq_map_plannedyear = df['Planned_Year'].value_counts(normalize=True).to_dict()
            df['Planned_Year_freq'] = df['Planned_Year'].map(freq_map_plannedyear)

            ##########
            #6PlannedWeek - Cyclic and Frequency Encoding
            ##########
            #Cyclical Encoding
            df['Planned_Week_sin'] = np.sin(2 * np.pi * df['Planned_Week'] / 12)
            df['Planned_Week_cos'] = np.cos(2 * np.pi * df['Planned_Week'] / 12)


            # Frequency Encoding #Based on the No Of Trips
            freq_map_plannedweek = df['Planned_Week'].value_counts(normalize=True).to_dict()
            df['Planned_Week_freq'] = df['Planned_Week'].map(freq_map_plannedweek)

            ##########
            #7IsWeekend - Frequency Encoding
            ##########
            freq_map_isweekend = df['IsWeekend'].value_counts(normalize=True).to_dict()
            df['IsWeekend_freq'] = df['IsWeekend'].map(freq_map_isweekend)
            logging.info("3.8 Cyclical & Frequency Based Features for DateTime Based columns created")
            print("Shape After Transformation - 3.8 - Deriving Cyclical & Frequency Features from columns created in 3.7:",df.shape)
        
################################################
#3.9 - Frequency Encoding for CustomerID,RelationID,CarrierID,OrderType
            print("Shape before Transformation - 3.9 - Frequency Encoding for Ids:",df.shape)
            # Step - 1 Frequency Encoding For CustomerID,
            freq_map_custid = df['CustomerID'].value_counts(normalize=True).to_dict()
            df['CustomerID_freq'] = df['CustomerID'].map(freq_map_custid)


            # Step - 2 Frequency Encoding For RelationID,
            freq_map_relationid = df['RelationID'].value_counts(normalize=True).to_dict()
            df['RelationID_freq'] = df['RelationID'].map(freq_map_relationid)


            # Step - 3 Frequency Encoding For CarrierID,
            freq_map_carrierid = df['CarrierID'].value_counts(normalize=True).to_dict()
            df['CarrierID_freq'] = df['CarrierID'].map(freq_map_carrierid)

            # Step - 4 Frequency Encoding For Order type,
            freq_map_ordertype = df['Order type'].value_counts(normalize=True).to_dict()
            df['OrderType_freq'] = df['Order type'].map(freq_map_ordertype)
            logging.info("3.9 - Frequency Encoding for CustomerID,RelationID,CarrierID,OrderType created successfully.")
            print("Shape After Transformation - 3.9 - Frequency Encoding for Ids:",df.shape)
            
################################################
#3.10 - Interaction Based Features
            print("Shape Before Transformation - 3.10 - Interaction Based Features:",df.shape)
            df['Carrier_Relation_Interaction'] = df['CarrierID_freq'] * df['RelationID_freq']
            df['Carrier_Relation_Customer_Interaction'] = df['CarrierID_freq'] * df['RelationID_freq'] * df['CustomerID_freq']
            df['Customer_Relation_Interaction'] = df['CustomerID_freq'] * df['RelationID_freq']
            df['Customer_Carrier_Interaction'] = df['CustomerID_freq'] * df['CarrierID_freq']
            logging.info("3.10 - Interaction Based Features created successfully.")
            print("Shape After Transformation - 3.10 - Interaction Based Features:",df.shape)
            
        
################################################
#3.11 - Feature Based on CarrierId & RelationId Aggregation of Number of Orders
            print("Shape Before Transformation - 3.11 - Feature Based on CarrierId & RelationId Aggregation of Number of Orders:",df.shape)
            # Calculate the frequency of each CarrierID and RelationID combination
            order_frequency = df.groupby(['CarrierID', 'RelationID'])['NumberOfOrders'].sum() / len(df)

            # Convert the result to a DataFrame
            order_frequency = order_frequency.rename('Carrier_Relation_Order_Frequency').reset_index()

            # Merge the frequency encoding back to the original dataframe
            df = df.merge(order_frequency, on=['CarrierID', 'RelationID'], how='left')
            logging.info("3.11 - Feature Based on CarrierId & RelationId Aggregation of Number of Orders created successfully.")
            print("Shape After Transformation - 3.11 - Feature Based on CarrierId & RelationId Aggregation of Number of Orders:",df.shape)
        
################################################
#3.12 - Ranking Carriers based on Trip Freq
            print("Shape Before Transformation - 3.12 - Ranking Carriers based on Trip Freq:",df.shape)
            # Define the thresholds for ranking based on the Carried_Ord_Freq
            high_threshold = df['CarrierID_freq'].quantile(0.67)  # Top 33% (High)
            medium_threshold = df['CarrierID_freq'].quantile(0.33)  # Middle 33% (Medium)

            # Create a function to rank the carriers based on Carried_Ord_Freq
            def rank_carriers(freq):
                if freq >= high_threshold:
                    return 3  # High
                elif freq >= medium_threshold:
                    return 2  # Medium
                else:
                    return 1  # Low

            # Apply the rank function to the Carried_Ord_Freq column
            df['CarrierRank'] = df['CarrierID_freq'].apply(rank_carriers)
            logging.info("3.12 - Feature Based on ranking of carriers created successfully.")
            print("Shape After Transformation - 3.12 - Ranking Carriers based on Trip Freq:",df.shape)

################################################
#3.13 - Feature based on OrderDensity Per Hour
            print("Shape Before Transformation - 3.13 - Feature based on OrderDensity Per Hour:",df.shape)
            # Calculate Order Density by Hour
            order_density_hour = df['Planned_Hour'].value_counts(normalize=True).rename('Order_Density_Per_Hour')


            # Merge back into the original dataframe if needed
            df = df.merge(order_density_hour, left_on='Planned_Hour', right_index=True, how='left')
            logging.info("3.13 - Feature based on OrderDensity Per Hour created successfully.")
            print("Shape After Transformation - 3.13 - Feature based on OrderDensity Per Hour:",df.shape)
            
        
################################################
#3.14 - Dropping all ID based columns
            print("Shape before Transformation - 3.14 - Dropping All ID Based column:",df.shape)
            # Calculate Order Density by Hour
            columns_to_drop = ["Date","CarrierID","RelationID","Trip Nr","Order type","CustomerID","PlannedDateTime","ArrivedDateTime","Delay","Planned_Week"]
            df = df.drop(columns=columns_to_drop)
            logging.info("3.14 - Dropped all ID based columns successfully.")
            print("Shape before Transformation - 3.14 - Dropping All ID Based column:",df.shape)
            return df
        except Exception as e:
            raise CustomException(e, sys)


   #def get_data_transformation_pipeline(self):
#     '''
#     This function returns a pipeline for data transformation that combines:
#     - Target variable derivation
#     - Removing null values from PlannedDateTime
#     - Removing duplicate Trip Numbers
#     '''
#     try:
#         # Create a pipeline with methods as steps
#         pipeline = Pipeline([
#             ("remove_null_planned_time", self.remove_null_planned_time),
#             ("derive_target_variable", self.derive_target_variable),
#             ("remove_duplicates", self.remove_duplicate_trip_numbers)
#         ])
#         return pipeline
#     except Exception as e:
#         raise CustomException(e, sys)

    def initiate_data_transformation2(self,train_data_transformed_path,test_data_transformed_path) -> pd.DataFrame:
        '''
        This function initiates the data transformation using the pipeline
        '''
        try:
            logging.info(f"Reading path for CSV files for train and test")
            #Step-1: Loading the Train and Test data from the path provided
            train_df_transformed = pd.read_csv(train_data_transformed_path)
            test_df_transformed = pd.read_csv(test_data_transformed_path)

            train_df_transformed["PlannedDateTime"] = pd.to_datetime(train_df_transformed["PlannedDateTime"], format="%Y-%m-%d %H:%M:%S")
            train_df_transformed["ArrivedDateTime"] = pd.to_datetime(train_df_transformed["ArrivedDateTime"], format="%Y-%m-%d %H:%M:%S")
            

            test_df_transformed["PlannedDateTime"] = pd.to_datetime(test_df_transformed["PlannedDateTime"], format="%Y-%m-%d %H:%M:%S")
            test_df_transformed["ArrivedDateTime"] = pd.to_datetime(test_df_transformed["ArrivedDateTime"], format="%Y-%m-%d %H:%M:%S")
            
            
            print("Information of DataFrame")
            print(train_df_transformed.info())

            #Step-2: getting an object ready
            preprocessing_obj = self.get_data_transformer_object()
            train_df_final = self.preprocessing_obj(train_df_transformed)
            test_df_final = self.preprocessing_obj(test_df_transformed)
            
            # Step 3: Save the transformed data to CSV files
            logging.info("Saving transformed train and test datasets.")
            train_df_final.to_csv(
                self.data_transformation_config2.train_data_final_path, index=False, header=True
            )

            test_df_final.to_csv(
                self.data_transformation_config2.test_data_final_path, index=False, header=True
            )
            #Step 4: New Addition by Arnob - Dropping the Target Variable from Train and Test Sets finally created
            target_column_name = "TargetVariable"
            input_feature_train_df = train_df_final.drop(columns = [target_column_name],axis = 1)
            target_feature_train_df = train_df_final[target_column_name]

            input_feature_test_df = test_df_final.drop(columns = [target_column_name],axis = 1)
            target_feature_test_df = test_df_final[target_column_name]

            #Step 5: Our dataframe is all messed up, so we will conver it into an array(easier for ML models to work)
            #And , we will put the Target variable column at the very last
            train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]

            #Step 6: Saving the preprocessing object

              #we write this function save_object in utils
            save_object(
                #to save the pickle file
                file_path = self.data_transformation_config2.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("Transformation of the datasets into final CSV files is completed.")
            return (
                # Step 4: Returning the path of the final csv file paths for next step
                self.data_transformation_config2.train_data_final_path,
                self.data_transformation_config2.test_data_final_path,
                train_arr,
                test_arr,
                self.data_transformation_config2.preprocessor_obj_file_path,

            )
        except Exception as e:
            raise CustomException(e, sys)
        
        