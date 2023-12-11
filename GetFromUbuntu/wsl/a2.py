# import os
# # Enable oneDNN optimizations
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# # Now you can import and use TensorFlow
# import tensorflow as tf

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb
 
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn import metrics
import pandas as pd
import pymssql

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

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
SELECT * FROM BTCUSDT_1w 
"""

# Execute the query
cursor.execute(SQL_QUERY)

# Fetch all the records
records = cursor.fetchall()
df = pd.DataFrame(records,columns = ['Kline_open_time','Open_price','High_price','Low_price','Close_price','Volume','Kline_close_time','Quote_asset_volume','Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume'])
#df = pd.DataFrame(records,columns = ['Kline_open_time','Open_price','High_price','Low_price','Close_price','Volume'])
print('df-head start')
print(df.head())
print('df-head stop')
print(df)
# Print the records
# for r in records:
#   print(f"{r['Kline_open_time']}\t{r['Open_price']}\t{r['High_price']}")

# Close the connection
conn.close()
print('end');
# import pandas as pd
# from sqlalchemy import create_engine

# engine = create_engine('mssql+pyodbc://@DESKTOP-CBPDCAC/BINANCE_V19?driver=ODBC+Driver+17+for+SQL+Server')

# df = pd.read_sql('SELECT * FROM BTCUSDT_1h' 
#                  '', engine, index_col='Kline_open_time')

# print(df.head())

df['open-close']  = df['Open_price'] - df['Close_price']
df['low-high']  = df['Low_price'] - df['High_price']
df['target'] = np.where(df['Close_price'].shift(-1) > df['Close_price'], 1, 0)



# Load your data
# df = pd.read_csv('bitcoin_price.csv')

# Preprocess your data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close_price'].values.reshape(-1,1))

# Prepare your data for the LSTM model
look_back = 60
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build your LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Train your LSTM model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=1, batch_size=1, verbose=2)

# Make predictions
predicted_price = model.predict(X)
print(predicted_price)
predicted_price = scaler.inverse_transform(predicted_price)
print(predicted_price)

# Predict the next closing price
last_60_days = scaled_data[-60:]
last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
next_close_price = model.predict(last_60_days)
next_close_price = scaler.inverse_transform(next_close_price)

print(f"The next close price is predicted to be {next_close_price}")

print('end2')