# LSTM stock forcasting app - Joseph Trierweiler

# Imports
import os
import pandas_datareader as pdr
import warnings
import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
import absl.logging
from datetime import date
import csv

# Warning suppression
warnings.simplefilter(action='ignore', category=FutureWarning)
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# User parameters
STOCKS = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'NVDA', 'UNH', 'JNJ', 'FB', 'PG', 'XOM', 'V', 'JPM', 'MA', 'HD', 'CVX', 'PFE', 'ABBV', 'BAC', 'KO', 'COST', 'PEP', 'AVGO', 'LLY', 'MRK', 'WMT', 'TMO', 'CSCO', 'DIS', 'ABT', 'VZ', 'ACN', 'ADBE', 'INTC', 'MCD', 'CMCSA', 'CRM', 'WFC', 'BMY', 'QCOM', 'DHR', 'TXN', 'NKE', 'LIN', 'PM', 'UNP', 'AMD', 'RTX', 'NEE']
SIMULATE_PERFORMANCE = 0
GET_STOCK_CSV = 0
VERBOSE = 0

# Model parameters
EPOCHS = 25
METHODS = [1,2]
SIZES = [100,150]

# Simulation parameters
DAYS = 255
REGENERATE_MODELS_EVERY_X_DAYS = 85 # Pick a multiple of your input DAYS please :)
STARTING_CASH  = 5000

# Functions
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def getPredictedProfit(predictedProfit):
    return predictedProfit['predicted_profit']

# Get the stock information from Tiingo
if GET_STOCK_CSV:
    print("== Getting stock CSV's from Tiingo ==\n")

    # Variables
    key = '26be6a052b3fc4d0995f25cc8e182c5b4f477eb9'

    # Create STOCKS directory
    dir = 'STOCKS/'
    if os.path.isfile(dir):
        print("Creating " + dir)
        os.mkdir(dir)

    # Clear STOCKS directory
    else:
        print(dir + " Exists\nRemoving STOCK.csv files")
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    # Get STOCK.csv files
    for STOCK in STOCKS:
        print("Getting " + STOCK + ".csv")
        df = pdr.get_data_tiingo(STOCK, api_key=key, start='1/1/12')
        df.to_csv("STOCKS/" + STOCK + '.csv')
    print("\n== Complete ==")

# Simulation loop
if SIMULATE_PERFORMANCE:

    print("== Simulating model performance from " + str(DAYS) + " days ago to now ==\n")

    # Variables
    cash = STARTING_CASH
    previous_cash = cash
    stock1_name = ""
    stock1_shares = 0
    best_model_name = ""
    best_model_score = 1

    # Create MODELS directory if it doesn't already exist
    dir = 'MODELS/'
    if ~os.path.isfile(dir):
        print(dir + " Exists")
    else:
        print("Creating " + dir)
        os.mkdir(dir)
    
    # Create BEST_MODELS directory if it doesn't already exist
    dir = 'BEST_MODELS/'
    if ~os.path.isfile(dir):
        print(dir + " Exists")
    else:
        print("Creating " + dir)
        os.mkdir(dir)

    # Get Date
    print("Getting date")
    today = date.today()
    print("Today's date is " + str(today.month) + "/" + str(today.day) + "/" + str(today.year))

    for day in range(DAYS):

        today = DAYS - (day)
        print("== Starting simulation for stock prices from " + str(today) + " days ago ==\n")
        if ((day + REGENERATE_MODELS_EVERY_X_DAYS) % REGENERATE_MODELS_EVERY_X_DAYS == 0):
            print("== Regenerating Models ==")

            # Clear MODELS directory
            dir = 'MODELS/'
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
            
            # Clear BEST_MODELS directory
            dir = 'BEST_MODELS/'
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
            
            # Generate a model for each STOCK in STOCKS
            for STOCK in STOCKS:

                # Get stock data and format for Tensorflow
                dataframe = pandas.read_csv("STOCKS/" + STOCK + '.csv', usecols=[2], engine='python', skipfooter=today) # Reads stock closing prices from CSV
                dataset = dataframe.values
                dataset = dataset.astype('float32')
                scaler = MinMaxScaler(feature_range=(0,1))
                dataset = scaler.fit_transform(np.array(dataset).reshape(-1,1)) # Transforms USD to be a float between 0 and 1, preserving data

                # Split data for training versus testing
                train_size = int(len(dataset) * 0.75)
                test_size = len(dataset) - train_size
                train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
                train = train.reshape(len(train),1,1)
                test = test.reshape(len(test),1,1)
                trainX, trainY = create_dataset(train, 1)
                testX, testY = create_dataset(test, 1)

                best_model_name = ''
                best_model_score = 1
                new_best_model_score = 0

                # Try different methods for generating the best model
                for METHOD in METHODS:
                    for SIZE in SIZES:
                        name = STOCK + "-" + str(METHOD) + "-" + str(SIZE)
                        model = Sequential()
                        if METHOD == 1:
                            model.add(Bidirectional(LSTM(SIZE, activation='tanh')))
                            model.add(Dense(1))
                        if METHOD == 2:
                            model.add(LSTM(SIZE, activation='tanh', return_sequences=True))
                            model.add(LSTM(SIZE, activation='tanh', return_sequences=True))
                            model.add(LSTM(SIZE, activation='tanh'))
                            model.add(Dense(1))
                        model.compile(loss='mean_squared_error', optimizer='adam')
                        filepath = "MODELS/" + name + ".hdf5"  # unique file name that will include the epoch and the validation acc for that epoch
                        checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=VERBOSE, save_best_only=True, mode='min') # saves only the best ones
                        model.fit(trainX, trainY, epochs=EPOCHS, batch_size=64, verbose=VERBOSE, callbacks=[checkpoint])
                        new_best_model_score = model.evaluate(testX, testY, verbose=VERBOSE)
                        # print("NBMS: " + str(new_best_model_score))
                        # print("BMS: " + str(best_model_score))
                        if new_best_model_score < best_model_score:
                            best_model_name = name
                            best_model_score = new_best_model_score
                model = keras.models.load_model("MODELS/" + best_model_name + ".hdf5")
                print("The best model for " + STOCK + " is " + best_model_name + " with a MSE of " + str(best_model_score))
                model.save("BEST_MODELS/" + STOCK + ".hdf5")

            print("\n== Complete ==\n")

        print("Starting forcasting")

        pairings=[]
        if(stock1_shares > 0):
            dataframe2 = pandas.read_csv("STOCKS/" + stock1_name + '.csv', usecols=[2], engine='python', skipfooter=today)
            dataset = dataframe2.values
            dataset = dataset.astype('float32')
            print("Selling " + str(stock1_name) + " shares for " + str(dataset[-1]) + " a share")
            cash_from_selling = dataset[-1] * stock1_shares
            cash_from_selling = round(float(cash_from_selling), 2)
            cash = round(float(cash),2)
            print("Cash from selling: " + str(cash_from_selling) + " Cash from yesterday: " + str(cash))
            cash = cash_from_selling + cash
            profit = cash - previous_cash
            
            print("Cash Today: " + str(cash) + " Cash Yesterday: " + str(previous_cash))
            print("Profit: " + str(profit) + "\n")
            profit = round(float(profit), 2)
            dataframe4 = pandas.read_csv("STOCKS/" + stock1_name + '.csv', usecols=[1], engine='python', skipfooter=today)
            dataset4 = dataframe4.values

            # Create LOGS directory if it doesn't already exist
            row = ['Sold',stock1_name,stock1_shares,dataset4[-1],dataset[-1]]
            writer.writerow(row)
        for STOCK in STOCKS:
            dataframe = pandas.read_csv("STOCKS/" + STOCK + '.csv', usecols=[2], engine='python', skipfooter=today)
            dataset = dataframe.values
            dataset = dataset.astype('float32')
            scaler = MinMaxScaler(feature_range=(0,1))
            dataset = scaler.fit_transform(np.array(dataset).reshape(-1,1))
            train_size = int(len(dataset) * 0.75)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
            train = train.reshape(len(train),1,1)
            test = test.reshape(len(test),1,1)
            trainX, trainY = create_dataset(train, 1)
            testX, testY = create_dataset(test, 1)
            model = keras.models.load_model("BEST_MODELS/" + STOCK + ".hdf5")
            testX = np.concatenate((testX,[[dataset[-1]]]))
            prediction = model.predict(testX)
            prediction = scaler.inverse_transform(prediction)
            prediction = prediction.reshape(len(prediction),1)
            dataset = scaler.inverse_transform(dataset)
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

            yesterday_prediction = prediction[-2]
            today_actual = dataset[-1]
            today_prediction = prediction[-1]
            print("Stock value on day " + str(today) + " for " + STOCK + " is " + str(today_actual))
            
            print("Cash available: " + str(cash))
            shares_possible_to_buy = cash // today_actual
            print("Shares possible to buy: " + str(shares_possible_to_buy))

            today_prediction_delta = (today_prediction - today_actual)
            print("The forcasted delta of " + STOCK + " is " + str(today_prediction_delta))
            
            last_test = test[-1]

            # print("Len of prediciton list: " + str(len(prediction)) + " Len of actual test values: " + str(len(test)))
            N = len(test)
            sum = 0
            for i in range(N):
                    if i > 0:
                        sum = sum + abs((test[i] - prediction[i-2])/test[i])
            average_percentage_error = sum * (1/(N-1))
            print("MAPE: " + str(average_percentage_error))
            today_prediction_delta = (today_prediction - today_actual) * (1 - average_percentage_error)
            print("The forcasted delta of " + STOCK + " is " + str(today_prediction_delta) + " including MAPE")
            predicted_cash_after_selling = ( (cash - (today_actual * shares_possible_to_buy)) + ((today_prediction_delta + today_actual) *shares_possible_to_buy) ) # leftover cash from buying today, plus the cash gained from selling tomorrow
            print("Predicted cash after buying the stock: " + str(predicted_cash_after_selling))
            predicted_profit = predicted_cash_after_selling - cash
            print("Predicted Profit: " + str(predicted_profit) + "\n")
            pairings.append({'stock':STOCK,'predicted_profit':predicted_profit})
        pairings.sort(key=getPredictedProfit)
        print("Best stock to buy is: " + str(pairings[-1].get('stock')) + " at a predicted profit of " + str(pairings[-1].get('predicted_profit')))
        stock1_name = str(pairings[-1].get('stock'))
        dataframe1 = pandas.read_csv("STOCKS/" + stock1_name + '.csv', usecols=[2], engine='python', skipfooter=today)
        dataset = dataframe1.values
        dataset = dataset.astype('float32')
        stock1_shares = cash // dataset[-1]
        previous_cash = cash
        cash = cash - (dataset[-1] * stock1_shares)
        cash = round(float(cash), 2)
        print("Bought " + str(stock1_shares) + " of " + str(stock1_name) + " at the price " + str(dataset[-1]))
        print("Cash before buying: " + str(previous_cash))
        print("Cash after buying: " + str(cash) + "\n")

        dataframe3 = pandas.read_csv("STOCKS/" + stock1_name + '.csv', usecols=[1], engine='python', skipfooter=today)
        dataset3 = dataframe3.values

        if day==0:
            # Create LOGS directory if it doesn't already exist
            dir = 'LOGS/'
            if ~os.path.isfile(dir):
                print(dir + " Exists")
            else:
                print("Creating " + dir)
                os.mkdir(dir)
            f = open(dir + "log.csv", 'w')
            writer = csv.writer(f)
        row = ['Bought',stock1_name,stock1_shares,dataset3[-1],dataset[-1]]
        writer.writerow(row)
        print("\n== Complete ==\n")
    print("\n== Complete ==\n")

# Generate models for today's prices
else:
    print("== Summing simulation performance ==\n")

    # Get trade type
    dataframe4 = pandas.read_csv("LOGS/log.csv", usecols=[0], engine='python', header=None)
    datasetStockType = dataframe4.values

    # Get stock names
    dataframe = pandas.read_csv("LOGS/log.csv", usecols=[1], engine='python', header=None)
    datasetStockNames = dataframe.values

    # Get stock shares
    dataframe3 = pandas.read_csv("LOGS/log.csv", usecols=[2], engine='python', header=None)
    datasetStockShares = dataframe3.values

    sum = STARTING_CASH
    today = DAYS
    print("Starting cash: " + str(sum))
    for i in range(len(dataframe)):
        if i==0:
            today = DAYS
            print("Days ago: " + str(today))
            dataframe2 = pandas.read_csv("STOCKS/" + str(datasetStockNames[i]).replace('[\'','').replace('\']','') + '.csv', usecols=[2], engine='python', skipfooter=today)
            datasetStockPrice = dataframe2.values
            datasetStockPrice = datasetStockPrice.astype('float32')
            print("Bought: " + str(datasetStockNames[i]).replace('[\'','').replace('\']','') + " Shares: " + str(float(str(datasetStockShares[i]).replace('[\'[','').replace('.]\']',''))))
            print("Price per share: " + str(datasetStockPrice[-1]))
            sum = sum - float(str(datasetStockShares[i]).replace('[\'[','').replace('.]\']','')) * datasetStockPrice[-1]
            sum = round(float(sum), 2)
            print("Cash: " + str(sum) + "\n")
        elif i > 0:
            today = DAYS - ((i+1) // 2)
            print("Days ago: " + str(today))
            dataframe2 = pandas.read_csv("STOCKS/" + str(datasetStockNames[i]).replace('[\'','').replace('\']','') + '.csv', usecols=[2], engine='python', skipfooter=today)
            datasetStockPrice = dataframe2.values
            datasetStockPrice = datasetStockPrice.astype('float32')

            if str(datasetStockType[i]).replace('[\'','').replace('\']','') == 'Bought':
                print("Bought: " + str(datasetStockNames[i]).replace('[\'','').replace('\']','') + " Shares: " + str(float(str(datasetStockShares[i]).replace('[\'[','').replace('.]\']',''))))
                print("Price per share: " + str(datasetStockPrice[-1]))
                sum = sum - float(str(datasetStockShares[i]).replace('[\'[','').replace('.]\']','')) * datasetStockPrice[-1]
                sum = round(float(sum), 2)
                print("Cash: " + str(sum))
            else:
                print("Sold: " + str(datasetStockNames[i]).replace('[\'','').replace('\']','') + " Shares: " + str(float(str(datasetStockShares[i]).replace('[\'[','').replace('.]\']',''))))
                print("Price per share: " + str(datasetStockPrice[-1]))
                sum = sum + float(str(datasetStockShares[i]).replace('[\'[','').replace('.]\']','')) * datasetStockPrice[-1]
                sum = round(float(sum), 2)
                print("Cash: " + str(sum) + "\n")
        
            
    sum = sum + float(str(datasetStockShares[-1]).replace('[\'[','').replace('.]\']','')) * datasetStockPrice[-1]
    print("Final cash: " + str(sum))
    simulation_profit = sum - STARTING_CASH
    print("The simulation made a profit of: " + str(simulation_profit))

    index_sum = 0
    for STOCK in STOCKS:
        today = DAYS
        dataframe2 = pandas.read_csv("STOCKS/" + STOCK + '.csv', usecols=[2], engine='python', skipfooter=today)
        datasetStockPrice = dataframe2.values
        datasetStockPrice = datasetStockPrice.astype('float32')
        index_sum = index_sum + datasetStockPrice[-1]
    previous_index = index_sum / len(STOCK)
    print("STOCKS index " + str(DAYS) + " days ago " + str(previous_index))

    index_sum = 0
    for STOCK in STOCKS:
        today = 0
        dataframe2 = pandas.read_csv("STOCKS/" + STOCK + '.csv', usecols=[2], engine='python', skipfooter=today)
        datasetStockPrice = dataframe2.values
        datasetStockPrice = datasetStockPrice.astype('float32')
        index_sum = index_sum + datasetStockPrice[-1]
    index_today = index_sum / len(STOCK)
    print("STOCKS index today: " + str(index_today))

    print("\n== Complete ==")