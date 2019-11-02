


from flask import Flask,render_template,request,url_for
import pandas as pd 
import numpy as np
import requests
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


@app.route("/predict/<sample>")
def predict(sample):
    
    sample = int(sample)
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

    features_set_single = features_set_2[sample]
    features_set_single = np.reshape(features_set_single,(1,200,3))

    loaded_model._make_predict_function()
    prediction = loaded_model.predict(features_set_single)
    result = np.where(prediction[0] == np.amax(prediction[0]))
    predicted_class = result[0][0]
    
    if (predicted_class == 0):
        predicted_class_activity = 'Downstairs'
    if (predicted_class == 1):
        predicted_class_activity = 'Jogging'
    if (predicted_class == 2):
        predicted_class_activity = 'Sitting'
    if (predicted_class == 3):
        predicted_class_activity = 'Standing'    
    if (predicted_class == 4):
        predicted_class_activity = 'Upstairs'
    if (predicted_class == 5):
        predicted_class_activity = 'Walking'
        
    golden=labels_2[sample]
    golden_label = np.where(golden == np.amax(golden))
    label = golden_label[0][0]
    
    if (label == 0):
        label_activity = 'Downstairs'
    if (label == 1):
        label_activity = 'Jogging'
    if (label == 2):
        label_activity = 'Sitting'
    if (label == 3):
        label_activity = 'Standing'    
    if (label == 4):
        label_activity = 'Upstairs'
    if (label == 5):
        label_activity = 'Walking'
    
    return_string = "Predicted class=" + predicted_class_activity + "; Expected class=" + label_activity
    return return_string






@app.route("/predict_result", methods=["GET", "POST"])
def my_form_post():
    
    if request.method == 'POST':
       result = request.form['inputrecord']
    #return result
        
    record = int(result)
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

    features_set_single = features_set_2[record]
    features_set_single = np.reshape(features_set_single,(1,200,3))

    loaded_model._make_predict_function()
    prediction = loaded_model.predict(features_set_single)
    result = np.where(prediction[0] == np.amax(prediction[0]))
    predicted_class = result[0][0]
    
    if (predicted_class == 0):
        predicted_class_activity = 'Downstairs'
    if (predicted_class == 1):
        predicted_class_activity = 'Jogging'
    if (predicted_class == 2):
        predicted_class_activity = 'Sitting'
    if (predicted_class == 3):
        predicted_class_activity = 'Standing'    
    if (predicted_class == 4):
        predicted_class_activity = 'Upstairs'
    if (predicted_class == 5):
        predicted_class_activity = 'Walking'
        
    golden=labels_2[record]
    golden_label = np.where(golden == np.amax(golden))
    label = golden_label[0][0]
    
    if (label == 0):
        label_activity = 'Downstairs'
    if (label == 1):
        label_activity = 'Jogging'
    if (label == 2):
        label_activity = 'Sitting'
    if (label == 3):
        label_activity = 'Standing'    
    if (label == 4):
        label_activity = 'Upstairs'
    if (label == 5):
        label_activity = 'Walking'
    
    #return_string = "Predicted class=" + predicted_class_activity + "; Expected class=" + label_activity
    predict_string = "Predicted class=" + predicted_class_activity + "; Expected class=" + label_activity
#     prediction = "Predicted class=" + predicted_class_activity + "; Expected class=" + label_activity
#     requested = {'prediction':prediction}
#     session['prediction'] = prediction
#     return render_template("index.html", requested=requested)
    return render_template("index.html", predict_string=predict_string)
    #return return_string
    
    



# @app.route("/predict_result",methods=['post','Get'])
# def showprediction():
    
#     user_input = request.form
#     url = user_input['url']
    
#     requested_prediction = {'url':url }
#     session['url'] = url
    
#     return render_template("index.html", requested_prediction=requested_prediction)

if __name__ == '__main__':
    #app.run(debug=True, threaded=False)
    app.run(host="127.0.0.1",port=5000, debug=True)

































