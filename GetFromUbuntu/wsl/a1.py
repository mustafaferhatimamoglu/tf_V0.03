import os
# Enable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Now you can import and use TensorFlow
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mssql+pyodbc://@DESKTOP-CBPDCAC/BINANCE_V19?driver=ODBC+Driver+17+for+SQL+Server')

df = pd.read_sql('SELECT * FROM BTCUSDT_1h' 
                 '', engine, index_col='Kline_open_time')

print(df.head())