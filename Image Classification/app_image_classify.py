###################################
#### Import Relevant Libraries ####
###################################
from flask import Flask,render_template,request,url_for, jsonify, session
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.python.framework import ops
from flask_session import Session
import joblib

from image_classifier import classify_img

####################################
#### Initialize App and Session ####
####################################
app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

#############################
#### Load Trained Models ####
#############################
model_file = "Output/models/ten/LR.sav"
global graph
graph = ops.get_default_graph()
classifier = joblib.load(model_file)
feature_extractor = inception_resnet_v2.InceptionResNetV2(weights='imagenet', 
                                                          include_top=False, 
                                                          input_shape=(224, 224, 3))
####################
#### App Routes ####
####################
@app.route('/')
def index():
    """Home route."""
    return render_template('index.html', requested_img={})

@app.route("/upload_image", methods=['POST', 'GET'])
def upload():
    """Reads in user input and displays uploaded image to page."""

    user_input = request.form
    url = user_input['url']

    requested_img = {'url': url}
    session['url'] = url

    return render_template("index.html", requested_img=requested_img)

@app.route("/predict")
def predict():
    """Processes image and make classification."""
    
    url = session.get('url')
    pred_class_str = classify_img(url, feature_extractor, classifier)
    requested_img = {"url": url, "prediction": pred_class_str}

    session.clear()
    return render_template("index.html", requested_img=requested_img)

if __name__ == '__main__':
    app.run(host="127.0.0.1",port=5000, debug=False)