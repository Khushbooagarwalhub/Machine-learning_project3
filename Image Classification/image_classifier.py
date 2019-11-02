###################################
#### Import Relevant Libraries ####
###################################
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import requests
import cv2

########################
#### Classify Image ####
########################

def classify_img(img_url, feature_extractor, classifier):

    # Extract image from url
    response = requests.get(img_url)
    img = np.uint8(Image.open(BytesIO(response.content)))

    # Pre-Process image
    img = img/255                           # rescale
    img = cv2.resize(img, (224, 224))       # resize
    img_arr = np.expand_dims(img, axis=0)   # increase dimensions to 4

    # Extract image features and reshape output into shape expected 
    # by classifier
    features = feature_extractor.predict(img_arr)
    features_shaped = np.reshape(features, (1, 5*5*1536)) 

    # Predict image class
    prediction = classifier.predict(features_shaped)

    # MAP CLASS INDEX TO CLASS NAME
    ### Read in csv of class indices and names
    class_indices = pd.read_csv("Output/image_features/ten/class_indices.csv")
    del class_indices['Unnamed: 0']
    classes_dict = class_indices.to_dict()

    ### Find current class id and set prediction equal to 
    ### associated class name
    for key, value in classes_dict['Class'].items():
        if str(key) == str(prediction[0]):
            pred_class = value

    ### Reformat string
    split_string = pred_class.split("_")
    pred_class_str = " ".join(split_string).capitalize()

    return pred_class_str
