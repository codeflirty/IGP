# import libraries
import numpy as np
import yfinance as yf
from flask import Flask, request, jsonify, render_template

import os
import math
import numpy as np
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from prettytable import PrettyTable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# create app and load the trained Model
app = Flask(__name__)

# Route to handle HOME
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle PREDICTED RESULT
@app.route('/',methods=['POST'])
def predict():
  
    inputs = [] # declaring input array
    bank = request.form['bank']
    date1 = request.form['date1']
    date2 = request.form['date2']
    model = request.form['model']
    factor = request.form['factor']
    next = request.form['next']
    
    bankt = ""
    
    if(bank=="1"):
        bankt="SBIN.NS"
    if(bank=="2"):
        bankt="AXISBANK.NS"
    if(bank=="3"):
        bankt="HDFCBANK.NS"
    if(bank=="4"):
        bankt="ICICIBANK.NS"
    if(bank=="5"):
        bankt="KOTAKBANK.NS"
    
    data = yf.Ticker(bankt)
    data = data.history(start=date1, end=date2)['Close']
    
    if(factor=="0"):
        scaler = MinMaxScaler(feature_range=(0,1))
        data = scaler.fit_transform(np.array(data).reshape(-1,1))
    if(factor=="1" and len(data)>700 and str(data.index[-1])>'2021-01-01'):
        data = data[:'2019-12-31'].append(data['2021-01-01':])
        scaler = MinMaxScaler(feature_range=(0,1))
        data = scaler.fit_transform(np.array(data).reshape(-1,1))
    
    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    data = np.array(clean_dataset(pd.DataFrame(data)))

    # Create Dataset

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 100

    # Dataset

    X_train, y_train = create_dataset(data, time_step)

    if(model=="0"):
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
    
        # Forecasting
        nexta = int(next)
        x_input = data[len(data)-100:].reshape(1,-1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        output = []
        n_steps = 100
        i = 0
    
        while(i < nexta):
            if(len(temp_input) > 100):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps))
                yhat = model.predict(x_input)
                temp_input.extend(yhat.tolist())
                temp_input = temp_input[1:]
                output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps))
                yhat = model.predict(x_input)
                temp_input.extend(yhat.tolist())
                output.extend(yhat.tolist())
                i=i+1
         
        plt.figure()
        plt.rcParams["figure.figsize"] = (14,5)
        day_new = np.arange(1,len(data)+1)
        day_pred = np.arange(len(data)+1,len(data)+1+nexta)
        plt.plot(day_new,scaler.inverse_transform(data).reshape(-1))
        plt.plot(day_pred,scaler.inverse_transform(np.array(output).reshape(-1, 1)))
        plt.legend(["Historical Data", "Predictions"])
        plt.ylabel("Closing Price (INR)")
        plt.xlabel("Days")
        plt.savefig('static/images/S1.png')
    
    if(model=="1"):
        model = LinearRegression()
        model.fit(X_train, y_train)
    
        # Forecasting
        nexta = int(next)
        x_input = data[len(data)-100:].reshape(1,-1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        output = []
        n_steps = 100
        i = 0
    
        while(i < nexta):
            if(len(temp_input) > 100):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps))
                yhat = model.predict(x_input)
                temp_input.extend(yhat.tolist())
                temp_input = temp_input[1:]
                output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps))
                yhat = model.predict(x_input)
                temp_input.extend(yhat.tolist())
                output.extend(yhat.tolist())
                i=i+1
         
        plt.figure()
        plt.rcParams["figure.figsize"] = (14,5)
        day_new = np.arange(1,len(data)+1)
        day_pred = np.arange(len(data)+1,len(data)+1+nexta)
        plt.plot(day_new,scaler.inverse_transform(data).reshape(-1))
        plt.plot(day_pred,scaler.inverse_transform(np.array(output).reshape(-1, 1)))
        plt.legend(["Historical Data", "Predictions"])
        plt.ylabel("Closing Price (INR)")
        plt.xlabel("Days")
        plt.savefig('static/images/S1.png')
    
    return render_template('index.html', predicted_result = "Result")

if __name__ == "__main__":
    app.run(debug=True)
