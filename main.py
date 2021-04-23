import numpy as np
# import matplotlib as mlt
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load data

company = 'AAPL' # Apple
start = dt.datetime(2020, 1, 1)
end = dt.datetime(2021, 2, 1)

data = web.DataReader(company, 'yahoo', start, end)

# Preprocess data
# Since values are very much variable here we require it to compress
# it too small values to identify it easier.
scaler = MinMaxScaler(feature_range=[0,1])

# We are using only close values here, the value of stocks after 
# market has stopped.
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# how many days to look back in past for values
prediction_days = 60

# to store training data
x_train = []
y_train = []

# going from pred days to length of scaled data, count from 60th index
# last index of scaled data.
for x in range(prediction_days, len(scaled_data)):
    # Here in x we adding data for first 60 values, using this we will 
    # we will find 61th value and so on
    x_train.append(scaled_data[x-prediction_days:x, 0]) # 60 values
    y_train.append(scaled_data[x,0]) # 61st value

print(x_train)
# Convert our arrays to numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

# Adding 1 addtional dimention here
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train)

# Build the model
model = Sequential()

# Units and layers are layers of data set to be trained more the number
# more time it will take to train data. return_sequences true because
# LSTM does feedback operation to feed again data.
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Prediction of next closing value

model.compile(optimizer='adam', loss='mean_squared_error')

# Now fitting data epochs 25 means 24 times iterate data
model.fit(x_train, y_train, epochs=25)


''' Test Model accuracy on historical data '''
# load test data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']))

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)


''' Make predictions on test data '''

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the test prediction
plt.plot(actual_prices, color='red', label='Actual company price')
plt.plot(predicted_prices, color='blue', label='Predicted company price')
plt.title(f"{company} shares price")
plt.xlabel("Time")
plt.ylabel(f"{company} shares price")
plt.legend()
plt.show()


# Predicting future data

real_data = [model_inputs[len(model_inputs) - 1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

print(scaler.inverse_transform(real_data[-1]))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
