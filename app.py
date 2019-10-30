# from flask import Flask,render_template,request,url_for
# import pandas as pd 
# import numpy as np 

# # ML Packages
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.externals import joblib

# # NLP
# # from textblob import TextBlob 
# from sklearn.externals import joblib
# import numpy as np
# import pandas as pd
# import os
# from pandas import DataFrame
# import csv

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.utils import to_categorical

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Reshape
# from keras.layers import Conv2D, MaxPooling2D
# import sklearn
# import sklearn.datasets
# from sklearn import metrics
# from sklearn.metrics import classification_report
# from sklearn import preprocessing
# import seaborn as sns

# from scipy import stats
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.models import model_from_json
# from keras.models import load_model

# class NeuralNetwork(object):
#     pass
# app = Flask(__name__)

# @app.route('/')
# def index():
#    return render_template('index.html')


# @app.route("/predict")
# def predict():
    
#     data = pd.read_csv('WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1.csv')
#     #data


#     # Define column name of the label vector
#     LABEL = 'Encodedactivity'
#     # Transform the labels from String to Integer via LabelEncoder
#     label_encoded = preprocessing.LabelEncoder()
#     # Add a new column to the existing DataFrame with the encoded values
#     data[LABEL] = label_encoded.fit_transform(data['activity'].values.ravel())

#     #data_train = data[data['user_id'] <=28]
#     data_test =  data[data['user_id'] > 28]
#     #data_train = data[data['user_id'] >=8]
#     #data_test =  data[data['user_id'] < 8]
#     #x_train = data_train[['acc_x','acc_y','acc_z']]
#     x_test = data_test[['acc_x','acc_y','acc_z']]
#     #y_train = data_train[['Encodedactivity']]
#     y_test = data_test[['Encodedactivity']]
#     #y_train

#     scaler = MinMaxScaler(feature_range = (0, 1))
#     x_testing_scaled = scaler.fit_transform(x_test)


#     features_set_2 = []
#     labels_2 = []

#     for i in range(0, 263400, 200):
#     #for i in range(0, 263400, 50):
#         features_set_2.append([x_testing_scaled[i:i+200, 0],x_testing_scaled[i:i+200, 1],x_testing_scaled[i:i+200, 2]])
#         max_labels_perwindow = stats.mode(y_test["Encodedactivity"][i: i+200])[0][0]
#         labels_2.append(max_labels_perwindow)

#     labels_2 = to_categorical(labels_2)

#     features_set_2, labels_2 = np.array(features_set_2), np.array(labels_2)
#     features_set_2= np.reshape(features_set_2,(1317,200,3))


#     #filename = "static/finalized_model_lstm.sav"
#     #loaded_model = joblib.load(filename)

    
#     #json_file = open('model.json', 'r')
#     #loaded_model_json = json_file.read()
#     #json_file.close()
#     #loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     #loaded_model.load_weights('model.h5')
#     #print('Loaded model from disk')
    
#     loaded_model = load_model('static/model_full.h5')

#     #loaded_model.evaluate(
#     #    features_set_2, labels_2, verbose=2)
#     features_set_single = features_set_2[0]
#     features_set_single = np.reshape(features_set_single,(1,200,3))
#     prediction = loaded_model.predict(features_set_single)

#     return prediction
#     #return loaded_model_json


# if __name__ == '__main__':
#     #app.run(debug=True, threaded=False)
#     app.run(host="127.0.0.1",port=5000,threaded=False, debug=True)


from flask import Flask,render_template,request,url_for
import pandas as pd 
import numpy as np

from tensorflow.python.framework import ops

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
# import joblib
from sklearn.externals import joblib

import numpy as np
import pandas as pd
import os
from pandas import DataFrame
import csv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import sklearn
import sklearn.datasets
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

class NeuralNetwork(object):
    pass
app = Flask(__name__)

global graph
graph = ops.get_default_graph()
loaded_model = load_model('model_full.h5', compile=False)

@app.route('/')
def index():
   return render_template('index.html')


@app.route("/predict")
def predict():
    
    data = pd.read_csv('WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1.csv')
    
    LABEL = 'Encodedactivity'
    label_encoded = preprocessing.LabelEncoder()
    data[LABEL] = label_encoded.fit_transform(data['activity'].values.ravel())

    data_test =  data[data['user_id'] > 28]
    x_test = data_test[['acc_x','acc_y','acc_z']]
    y_test = data_test[['Encodedactivity']]

    scaler = MinMaxScaler(feature_range = (0, 1))
    x_testing_scaled = scaler.fit_transform(x_test)


    features_set_2 = []
    labels_2 = []

    for i in range(0, 263400, 200):
        features_set_2.append([x_testing_scaled[i:i+200, 0],x_testing_scaled[i:i+200, 1],x_testing_scaled[i:i+200, 2]])
        max_labels_perwindow = stats.mode(y_test["Encodedactivity"][i: i+200])[0][0]
        labels_2.append(max_labels_perwindow)

    labels_2 = to_categorical(labels_2)

    features_set_2, labels_2 = np.array(features_set_2), np.array(labels_2)
    features_set_2= np.reshape(features_set_2,(1317,200,3))

    features_set_single = features_set_2[0]
    features_set_single = np.reshape(features_set_single,(1,200,3))

    loaded_model._make_predict_function()
    prediction = loaded_model.predict(features_set_single)

    return str(prediction)

if __name__ == '__main__':
    #app.run(debug=True, threaded=False)
    app.run(host="127.0.0.1",port=5000, debug=True)

































