from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline
from datetime import datetime

application=Flask(__name__) # This basically gives the entry point where we need to execute

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') # THis generally searches for the template folder

@app.route('/predictdata',methods=['GET','POST'])


# we will be calling the custom data, created in predict_pipeline
def predict_datapoint():
    '''
    Predicting the Datapoint, form action in HTML will trigger it 
    '''
    if request.method=='GET':
        return render_template('home.html')# Input data fields will be present
    else:
        # Parse the date in the format "%d/%m/%Y"
        date_str = request.form.get('date')
        date_obj = datetime.strptime(date_str, '%d/%m/%Y').date()
        formatted_date = date_obj.strftime('%d/%m/%Y')
        print("Printing DateObj",date_obj)
        # Parse planned_date and arrival_date in the same format if needed
        planned_date_str = request.form.get('planned_date')
        planned_date_obj = datetime.strptime(planned_date_str, '%d/%m/%Y').date()
        formatted_planned_date= planned_date_obj.strftime('%d/%m/%Y')

        arrival_date_str = request.form.get('arrival_date')
        arrival_date_obj = datetime.strptime(arrival_date_str, '%d/%m/%Y').date()
        formatted_arrival_date= arrival_date_obj.strftime('%d/%m/%Y')

        # Parse the time (Planned Time and Arrival Time)
        planned_time_str = request.form.get('planned_time')
        planned_time_obj = datetime.strptime(planned_time_str, '%I:%M %p').strftime('%I:%M:%S %p')  # Convert to required format with seconds
        

        arrival_time_str = request.form.get('arrival_time')
        arrival_time_obj = datetime.strptime(arrival_time_str, '%I:%M %p').strftime('%I:%M:%S %p')  # Convert to required format with seconds

        #for POST - Own custom class, this will be created in predict pipeline too
        data=CustomData(
            date=formatted_date,
            transport_company=request.form.get('transport_company'),
            relation_name=request.form.get('relation_name'),
            relation_code=request.form.get('relation_code'),
            trip_nr=request.form.get('trip_nr'),
            order_number=request.form.get('order_number'),
            external_reference=request.form.get('external_reference'),
            order_type=request.form.get('order_type'),
            customer=request.form.get('customer'),
            planned_date=formatted_planned_date,
            planned_time=planned_time_obj,
            arrival_date=formatted_arrival_date,
            arrival_time=arrival_time_obj

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results,proba=predict_pipeline.predict(pred_df)
        print("after Prediction")
        print("Results",results[0])
        print("Results",proba)
        return render_template('home.html',results=results[0],proba=proba)#Output will be in the list format
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)   