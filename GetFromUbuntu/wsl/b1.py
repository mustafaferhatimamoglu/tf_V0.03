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

