from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline


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
        #for POST - Own custom class, this will be created in predict pipeline too
        data=CustomData(
            date=request.form.get('date'),
            transport_company=request.form.get('transport_company'),
            relation_name=request.form.get('relation_name'),
            relation_code=request.form.get('relation_code'),
            trip_nr=request.form.get('trip_nr'),
            order_number=request.form.get('order_number'),
            external_reference=request.form.get('external_reference'),
            order_type=request.form.get('order_type'),
            customer=request.form.get('customer'),
            planned_date=request.form.get('planned_date'),
            planned_time=request.form.get('planned_time'),
            arrival_date=request.form.get('arrival_date'),
            arrival_time=request.form.get('arrival_time')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])#Output will be in the list format
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)   