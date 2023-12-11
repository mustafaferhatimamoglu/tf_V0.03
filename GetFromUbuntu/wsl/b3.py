import os
# # Enable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# # Now you can import and use TensorFlow
import tensorflow as tf
#region My Code Region

# Enable Database
import pymssql
import pandas as pd
import numpy as np # i don't know shit

from sklearn.preprocessing import MinMaxScaler

# Connect to your database
conn = pymssql.connect(
  server='172.31.0.1',
  user='sa',
  password='sapass',
  database='BINANCE_V19',
  as_dict=True
)
# Create a cursor
cursor = conn.cursor()
# Define your SQL query
SQL_QUERY = """
SELECT 
                  BTCUSDT_1h.* 
				  ,RSI_6,RSI_12,RSI_24  
				  ,K,D,J
				  ,[OBV_6],[OBV_12],[OBV_24]
				  ,[CP_ClosePrice_1]      ,[CP_ClosePrice_6]      ,[CP_ClosePrice_12]      ,[CP_ClosePrice_24]
				  ,[CP_LowPrice_1]      ,[CP_LowPrice_6]      ,[CP_LowPrice_12]      ,[CP_LowPrice_24]
				  ,[CP_HighPrice_1]      ,[CP_HighPrice_6]      ,[CP_HighPrice_12]      ,[CP_HighPrice_24]

                  FROM BTCUSDT_1h 
                  inner join BTCUSDT_1h_RSI on   BTCUSDT_1h.Kline_close_time = BTCUSDT_1h_RSI.Time
                  inner join BTCUSDT_1h_KDJ on   BTCUSDT_1h.Kline_close_time = BTCUSDT_1h_KDJ.Time
                  inner join BTCUSDT_1h_OBV on   BTCUSDT_1h.Kline_close_time = BTCUSDT_1h_OBV.Time
                  inner join BTCUSDT_1h_CP on   BTCUSDT_1h.Kline_close_time = BTCUSDT_1h_CP.Time

				  order by Kline_close_time
				  offset 100 ROWS 
          --FETCH NEXT "+(ML_LearnCounter- ML_Offset- ML_Predict) +" ROWS ONLY
"""
# Execute the query
cursor.execute(SQL_QUERY)
# Fetch all the records
records = cursor.fetchall()
df = pd.DataFrame(records)
#,columns = ['Kline_open_time','Open_price','High_price','Low_price','Close_price','Volume','Kline_close_time','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume'])

print('df-head start')
print(df.head())
print('df-head stop')
print(df)

# Close the connection
conn.close()
print('Close Connection');

df['open-close']  = df['Open_price'] - df['Close_price']
df['low-high']  = df['Low_price'] - df['High_price']
df.drop(columns=['Kline_open_time'], inplace=True)
df.drop(columns=['Kline_close_time'], inplace=True)
#endregion

# Preprocess your data
scaler = MinMaxScaler()
# Fit the scaler to the features and transform
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print('df_scaled')
print(df_scaled)
print('end')

#----------------------------------
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# Assume 'Low_price' and 'High_price' are the columns you want to predict
target_cols = ['Low_price', 'High_price']

# Prepare the data for the LSTM model
X, y = [], []
for i in range(1, len(df_scaled)):
    X.append(df_scaled.iloc[i-1:i, :].values)
    y.append(df_scaled.loc[i, target_cols].values)
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], df_scaled.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(len(target_cols)))

# Train the LSTM model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=2, batch_size=1, verbose=2)

# Make predictions
predicted_values = model.predict(X)
# Create a new scaler for the target columns
scaler_target = MinMaxScaler()
scaler_target.fit(df[target_cols])

# Invert the scaling operation to get the original scale
predicted_values = scaler_target.inverse_transform(predicted_values)

# Print the predicted values
print(predicted_values)

# Number of future steps to predict
future_steps = 100
# Predict the next 100 values
last_values = df_scaled[-future_steps:, :len(target_cols)]
for _ in range(100):
    # Reshape the last_values
    last_values_reshaped = np.reshape(last_values, (1, last_values.shape[0], last_values.shape[1]))
    
    # Predict the next value
    next_value = model.predict(last_values_reshaped)
    
    # Append the next_value to the last_values
    last_values = np.append(last_values, next_value, axis=0)
    
    # Remove the first value of last_values
    last_values = last_values[1:]

# Invert the scaling operation to get the original scale
next_100_values = scaler_target.inverse_transform(last_values)

# Print the next 100 values
print(next_100_values)