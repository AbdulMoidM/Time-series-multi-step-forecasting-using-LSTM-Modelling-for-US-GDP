import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
# import quandl
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Recession:
    def __init__(self, name, start_date, end_date):
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.training_input = self.build_training_input(12)
        self.training_output = self.build_training_output(3)

    def build_training_input(self, month):
        # for training input date we take month number from parameter that gives the number of months before start of recession
        # building all the revelant data into 2 dimensional array format to use LTSM algorithm
        training_input = []

        # adding snp data
        current_date = self.start_date - relativedelta(months=+month)
        training_snp = []
        while (current_date < self.start_date):
            training_snp.append(snp[snp['Date'] >= current_date]['Close'].iloc[0])
            current_date = current_date + relativedelta(months=+1)
        training_input.append(training_snp)
        # print (len(training_snp))

        # adding employee data
        current_date = self.start_date - relativedelta(months=+month)
        training_employee = []
        while (current_date < self.start_date):
            training_employee.append(employee[employee['Date'] >= current_date]['Value'].iloc[0])
            current_date = current_date + relativedelta(months=+1)
        training_input.append(training_employee)
        # print (len(training_employee))

        # adding treasury data
        current_date = self.start_date - relativedelta(months=+month)
        training_treasury = []
        while (current_date < self.start_date):
            temp_treasury = treasury[treasury['Date'] >= current_date]
            training_treasury.append((temp_treasury['SVENY07'].iloc[0]) - (temp_treasury['SVENY01'].iloc[0]))
            current_date = current_date + relativedelta(months=+1)
        training_input.append(training_treasury)
        # print (len(training_treasury))

        # adding gdp data
        current_date = self.start_date - relativedelta(months=+month)
        training_gdp = []
        while (current_date < self.start_date):
            training_gdp.append(gdp[gdp['Date'] >= current_date]['Value'].iloc[0])
            current_date = current_date + relativedelta(months=+1)
        training_input.append(training_gdp)
        # print (len(training_gdp))

        # print ("----------")

        # return the 2 dimensional list
        return training_input
    
    def build_training_output(self, month):
        # for training output date we take month number from parameter that gives the number of months after start of recession
        # building all the revelant data into 2 dimensional array format to use LTSM algorithm
        training_output = []

        # adding gdp data as it is the only output we are concerned with
        current_date = self.start_date + relativedelta(months=+1)
        training_gdp = []
        while (current_date <= self.start_date + relativedelta(months=+month)):
            training_gdp.append(gdp[gdp['Date'] >= current_date]['Value'].iloc[0])
            current_date = current_date + relativedelta(months=+1)
        training_output.append(training_gdp)

        # return the 2 dimensional list
        return training_output

# method to convert unformatted list to formatted list
def ltsm_formatter(unformatted):
    formatted = []
    for k in range(len(unformatted)):
        temp_1 = []
        for i in range(len(unformatted[k][0])):
            temp_2 = []
            for j in range(len(unformatted[k])):
                temp_2.append(unformatted[k][j][i])
            temp_1.append(temp_2)
        formatted.append(temp_1)
    return formatted

# creating pandas dataframe from the file
snp = pd.read_csv('snp.csv')
# employee data
employee = pd.read_csv('employee.csv')
# quandl.get("FRED/USGOVT").to_csv('employee.csv')
# treasury data
treasury = pd.read_csv('treasury.csv')
# quandl.get("FED/SVENY").to_csv('treasury.csv')
# output data, GDP
gdp = pd.read_csv('gdp_changed_smooth.csv')
# quandl.get("FRED/GDP").to_csv('gdp.csv')

# converting the data column to datetime format
snp['Date'] = pd.to_datetime(snp['Date'], format="%Y-%m-%d")
employee['Date'] = pd.to_datetime(employee['Date'], format="%Y-%m-%d")
treasury['Date'] = pd.to_datetime(treasury['Date'], format="%Y-%m-%d")
gdp['Date'] = pd.to_datetime(gdp['Date'], format="%Y-%m-%d")

# # smooth the gdp data
# start_date = gdp['Date'][0]
# end_date = gdp['Date'][len(gdp)-1]
# current_date = start_date
# next_date = current_date + relativedelta(months=+3)
# while ((current_date < end_date) and (next_date <= end_date)):
#     # delta value i.e. the change in price
#     d_value = (gdp[gdp['Date'] == next_date]['Value'].iloc[0] - gdp[gdp['Date'] == current_date]['Value'].iloc[0])/3
#     current_price = gdp[gdp['Date'] == current_date]['Value'].iloc[0]
#     current_date = current_date + relativedelta(months=+1)
#     while (current_date < next_date):
#         gdp.loc[len(gdp)] = [current_date, current_price + d_value]
#         current_date = current_date + relativedelta(months=+1)
#         current_price = current_price + d_value
#     next_date = next_date + relativedelta(months=+3)

# sorting the df by date
gdp.sort_values(by=['Date'], inplace=True)

# reset the indexing messed up by sorting
gdp.reset_index(drop='True', inplace=True)

# # getting the change in gdp
# for i in range(len(gdp)-1, 0, -1):
#     gdp['Value'][i] = gdp['Value'][i-1] - gdp['Value'][i]

# gdp.to_csv('gdp_changed_smooth.csv')

# data obtained from wikipedia: https://en.wikipedia.org/wiki/List_of_recessions_in_the_United_States
# lists that store past recession name, start date and end date of corresponding recession
recession_name = ['Recession of 1969–70', '1973–75 recession', '1980 recession', '1981–1982 recession', 
                'Early 1990s recession in the United States', 'Early 2000s recession', 'Great Recession']
recession_start = ['1969-12-01', '1973-11-01', '1980-01-01', '1981-07-01', '1990-07-01', '2001-03-01', '2007-12-01']
recession_end = ['1970-11-28', '1975-03-28', '1980-07-28', '1982-11-28', '1991-03-28', '2001-11-28', '2009-06-28']

# for corona
corona = []
corona_start_date = datetime.strptime("2019-02-01", '%Y-%m-%d')
corona_end_date = datetime.strptime("2020-02-01", '%Y-%m-%d')

# adding snp data
current_date = corona_start_date
training_snp = []
while (current_date < corona_end_date):
    training_snp.append(snp[snp['Date'] >= current_date]['Close'].iloc[0])
    current_date = current_date + relativedelta(months=+1)
corona.append(training_snp)

# adding employee data
current_date = corona_start_date
training_employee = []
while (current_date < corona_end_date):
    training_employee.append(employee[employee['Date'] >= current_date]['Value'].iloc[0])
    current_date = current_date + relativedelta(months=+1)
corona.append(training_employee)

# adding treasury data
current_date = corona_start_date
training_treasury = []
while (current_date < corona_end_date):
    temp_treasury = treasury[treasury['Date'] >= current_date]
    training_treasury.append((temp_treasury['SVENY07'].iloc[0]) - (temp_treasury['SVENY01'].iloc[0]))
    current_date = current_date + relativedelta(months=+1)
corona.append(training_treasury)

# adding gdp data
current_date = corona_start_date
training_gdp = []
while (current_date < corona_end_date):
    training_gdp.append(gdp[gdp['Date'] >= current_date]['Value'].iloc[0])
    current_date = current_date + relativedelta(months=+1)
corona.append(training_gdp)

temp_list = []
corona = temp_list.append(corona)
corona = temp_list

corona = ltsm_formatter(corona)

# print (corona)

# print (len(recession_name))
# print (len(recession_start))
# print (len(recession_end))

# list to hold all the recession objects
recessions = []
# creating object for the recessions and storing it in the recessions list
for i in range(len(recession_name)):
    # getting and parsing the relevant start and end date for recession
    start_date = datetime.strptime(recession_start[i], "%Y-%m-%d")
    end_date = datetime.strptime(recession_end[i], "%Y-%m-%d")
    # creating the object with necessary data
    new_recession = Recession(recession_name[i], start_date, end_date)
    # adding the recession to the list
    recessions.append(new_recession)

# testing
# for recession in recessions:
#     print ("{}, {}, {}".format(recession.name, recession.start_date, recession.end_date))
#     print ("-----")

# print ("{}, {}, {}".format(recessions[0].name, recessions[0].start_date, recessions[0].end_date))
# print (recessions[0].training_input)
# print (recessions[0].training_output)

# print (treasury['SVENY07'].iloc[0])
# print(recessions[0].start_date - relativedelta(months=+12))

# using first 5 recessions as training set
recessions_training = []
recessions_testing = []

recessions_training.append(recessions[0])
recessions_training.append(recessions[6])
recessions_training.append(recessions[1])
recessions_training.append(recessions[3])
recessions_training.append(recessions[5])

recessions_testing.append(recessions[2])
recessions_testing.append(recessions[4])

# initializing the training input and output
training_input = []
training_output = []
for recession in recessions_training:
    training_input.append(recession.training_input)
    training_output.append(recession.training_output)

# initializing the testing input and output
testing_input = []
testing_output = []
for recession in recessions_testing:
    testing_input.append(recession.training_input)
    testing_output.append(recession.training_output)

# brining the 3 dimensional arrays in required format for ltsm
training_input = ltsm_formatter(training_input)
training_output = ltsm_formatter(training_output)
testing_input = ltsm_formatter(testing_input)
testing_output = ltsm_formatter(testing_output)

# print (training_input)
# print (training_output)
# print (testing_input)
# print (testing_output)

# evaluate one or more GDP forecasts against expected values
def evaluate_forecasts(actual, predicted):#actual is 2d array(each row is output for one sample), and predicted is 2d array
	scores = list()
	q1=np.quantile(actual, 0.25)
	q3=np.quantile(actual, 0.75)
	qDiff=q3-q1
	# calculate an RMSE score for each month of GDP predicted
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i]) #first first i elements of all rows/samples
		# calculate rmse
		rmse = sqrt(mse)/qDiff
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))

	score=score/qDiff
	
	return score, scores
 
# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
 

 
# train the model
def build_model(train_x, train_y):
	
	# define parameters
	verbose, epochs, batch_size = 0, 2100, 4
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	#train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
 
# make a forecast
def forecast(model, history):
	
	# reshape into [1, n_input, n]
	input_x = history.reshape((1, history.shape[0], history.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat
 
# evaluate a single model
def evaluate_model(train_x,train_y, test_x,test_y):
	# fit model
	model = build_model(train_x, train_y)
	# history is a list of all data in training
	
        
	# walk-forward validation over each month predicted
	predictions = list()
    
	for history in test_x:
		# predict the week
        #print(history)
		yhat_sequence = forecast(model, history)
		# store the predictions
        #print(yhat_sequence)
		predictions.append(yhat_sequence)
        
	# evaluate predictions days for each week
	predictions = array(predictions)
    
	score, scores = evaluate_forecasts(test_y[:, :, 0], predictions)
	return score, scores, predictions
 
# i get recession_training_input and recession_training_output



trainingIn=array(training_input)
trainingOut=array(training_output)
testIn=array(testing_input)
testOut=array(testing_output)


scoreSum=0
scoresSum=[0,0,0]
predictionsSum=[0,0,0]
for i in range(10):
    score, scores,predictions = evaluate_model(trainingIn, trainingOut, testIn, testOut)
    print(predictions)
    scoreSum+=score/2
    scores=array(scores)
    
    for x in range(len(scores)):
        scoresSum[x]+=scores[x]/2
    for z in range(2):
        for y in range(3):
            predictionsSum[y]+=predictions[z][y][0]/10


summarize_scores('lstm', scoreSum, scoresSum)




# plot scores
plotArray=list()
firstArray=list()
print(testOut.shape[1])
for i in range(testOut.shape[1]):
    firstArray.append(testOut[0][i][0])
secondArray=list()
for i in range(len(predictionsSum)):
    secondArray.append(predictionsSum[i])

plotArray.append(array(firstArray))
plotArray.append(array(secondArray))
plotArray=array(plotArray)

print(plotArray)
months = ['month1', 'month2', 'month3']
pyplot.plot(months, plotArray[0], months, plotArray[1], marker='o', label='3 month prediction')
pyplot.show()






