from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline
from datetime import datetime
import os


application=Flask(__name__) # This basically gives the entry point where we need to execute

app = application

########################### Starting: Part required for uploading file
UPLOAD_FOLDER = os.path.abspath('uploads')
PROCESSED_FOLDER = os.path.abspath('processed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created directory: {UPLOAD_FOLDER}")
else:
    print(f"Directory already exists: {UPLOAD_FOLDER}")

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)
    print(f"Created directory: {PROCESSED_FOLDER}")
else:
    print(f"Directory already exists: {PROCESSED_FOLDER}")
########################### Ending: Part required for uploading file

@app.route('/')
def index():
    return render_template('home.html') # THis generally searches for the template folder

@app.route('/upload',methods=['GET','POST'])


# we will be calling the custom data, created in predict_pipeline
def predict_datapoint():
    '''
    Predicting the Datapoint, form action in HTML will trigger it 
    '''
    if request.method=='GET':
        return render_template('home.html')# Input data fields will be present
    else:
        #request.method is POST
        if 'file' not in request.files:
            return "No file part in the request"
        file = request.files['file']
        
        
        if file.filename == '':
            return "No selected file"
        if file and file.filename.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(f"Saving uploaded file to: {file_path}")
            

            # print(file_path)
            
            file.save(file_path)

            df = pd.read_csv(file_path)
            #orignal_df = df.copy()

        #print("############### Good Job ! Getting this Far! ###############")
        #print("#Step-1: We have accepted the Data - in the prescribed format")
        # Parse the date in the format "%d/%m/%Y"
        #date_str = request.form.get('date')
        df["Date"] = datetime.strptime(df["Date"], '%d/%m/%Y').date()
        df["Date"] = df["Date"].strftime('%d/%m/%Y')
    
        # Parse planned_date and arrival_date in the same format if needed
        #planned_date_str = request.form.get('planned_date')
        #planned_date_obj = datetime.strptime(planned_date_str, '%d/%m/%Y').date()
        #formatted_planned_date= planned_date_obj.strftime('%d/%m/%Y')
        df["Planned Date"] = datetime.strptime(df["Planned Date"], '%d/%m/%Y').date()
        df["Planned Date"] = df["Planned Date"].strftime('%d/%m/%Y')

        #arrival_date_str = request.form.get('arrival_date')
        #arrival_date_obj = datetime.strptime(arrival_date_str, '%d/%m/%Y').date()
        #formatted_arrival_date= arrival_date_obj.strftime('%d/%m/%Y')
        #df["arrival_date"] = datetime.strptime(df["arrival_date"], '%d/%m/%Y').date()
        #df["arrival_date"] = df["arrival_date"].strftime('%d/%m/%Y')

        # Parse the time (Planned Time and Arrival Time)
        #planned_time_str = request.form.get('planned_time')
        df["Planned Time"] = datetime.strptime(df["Planned Time"], '%I:%M %p').strftime('%I:%M:%S %p')  # Convert to required format with seconds
        #df["planned_time"] = datetime.strptime(df["planned_time"], '%I:%M %p').strftime('%I:%M:%S %p')
    
        

        #arrival_time_str = request.form.get('arrival_time')
        #arrival_time_obj = datetime.strptime(arrival_time_str, '%I:%M %p').strftime('%I:%M:%S %p')  # Convert to required format with seconds
        #df["arrival_time"] = datetime.strptime(df["arrival_time"], '%I:%M %p').strftime('%I:%M:%S %p')  # Convert to required format with seconds
        #df["arrival_time"] = datetime.strptime(df["arrival_time"], '%I:%M %p').strftime('%I:%M:%S %p')
        #for POST - Own custom class, this will be created in predict pipeline too
        #data=CustomData(
            #date=df["date"],
            #transport_company=df["transport_company"],
            #relation_name=df["relation_name"],
            #relation_code=df["relation_code"],
            #trip_nr=df["trip_nr"],
            #order_number=df["order_number"],
            #external_reference=df["external_reference"],
            #order_type=df["order_type"],
            #customer=df["customer"],
            #planned_date=df["planned_date"],
            #planned_time=df["planned_time"]
            #arrival_date=formatted_arrival_date,
            #arrival_time=arrival_time_obj

        #)
        #pred_df=data.get_data_as_data_frame()

        # Renaming columns columns
       # Column renaming dictionary
        # Column renaming dictionary
        #column_mapping = {
            #"date": "Date",
            #"transport_company": "Transport Company"
        #}
        #column_mapping = {
                #"Date": [self.date],
                #"Transport Company": [self.transport_company],
                #"RelationName":[self.relation_name],
                #"RelationCode":[self.relation_code],
                #"Trip Nr":[self.trip_nr],
                #"Order Number":[self.order_number],
                #"External reference":[self.external_reference],
                #"Order type":[self.order_type],
                #"Customer":[self.customer],
                #"Planned Date":[self.planned_date],
                #"Planned Time":[self.planned_time]
                #"Arrival Date":[self.arrival_date],
                #"Arrival Time":[self.arrival_time],
                #}        

        # Rename columns
        #df = df.rename(columns=column_mapping)

        #print(df)
        pred_df = df
        print(("########Step-2: Please view the captured DataFrame:"))
        print(pred_df)
        print("########Step-3: Havent Started Prediction Yet")

        predict_pipeline=PredictPipeline()
        print("########Step-4: Mid Prediction - Prediction Object Created")
        results,proba=predict_pipeline.predict(pred_df)
        print(results)
        print(proba)
        #Rounding off the probability of 1
        #prob_of_1 = proba[0, 1] #takes the second element in row 1 from the array
        #rounded_prob_of_1 = round(prob_of_1,2)

        pred_df["Delay_Percentage"] = proba[:,1]*100 #Converting Probability of class 1 to percentage
        pred_df["Status"] = (df["Delay_Percentage"]>=50).map({True: "Delayed", False: "On Time"})
        #Mapping the FInal Prediction 1= Delayed, 0 =OnTime
        # Use an if-else statement
        # Use map to convert 0 -> "OnTime" and 1 -> "Delayed"
        #mapped_results = list(map(lambda x: "OnTime" if x == 0 else "Delayed", results))

        print(pred_df)
        #print(probability_of_1)
        #Output is customeised only for 1 result
        print("########Step-8: Prediction Completed - Please find the results below")
        print("########Result-1:Prediction Results for the record:",mapped_results[0])
        print("########Result-2:Probability for Prediction Results for the record:",rounded_prob_of_1)
        
        processed_file_name = f"results_{file.filename}"
        processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_file_name)

        print(f"Processed file will be saved to: {processed_file_path}")
        pred_df.to_csv(processed_file_path, index=False)

        message = f"The final CSV file has been saved in the processed folder as: {processed_file_name}"

        return render_template('results.html', file_name = processed_file_name ,message = message)
    
        
    
    @app.route('/download/<filename>')
    def download_file(filename):
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        print(f"Attempting to download file from: {file_path}")  # Debugging
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
        return send_file(file_path, as_attachment=True)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)   