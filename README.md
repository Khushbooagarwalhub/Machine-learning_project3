# project_3
# Human Action Recognition (HAR)
Topic -
Machine Learning techniques in sensor based human activity recognition and understanding human actions using still images.
Folder for sensor based HAR -"sensor based HAR"

Folder for image classification - "Image Classification"

## Sensor-Based HAR
Jupyter notebooks - "DNN_model.ipynb"
                     "LSTM_model.ipynb"
                     "exploratory1.ipynb"
Resources folder - "WISDM_ar_latest"
Data source for sensor based HAR - WISDM dataset

http://www.cis.fordham.edu/wisdm/dataset.php



     Citation - "Activity Recognition using cell phone Accelerometers - Jennifer R Kwapisz,Gary M.Weiss,Samuel A.Moore

The source consist of raw time series data and transformed dataset.



The raw timeseries data consist of -
Raw time series data 
<br>
Number of examples 
1,098,207
<br>
Number of attributes – 6
<br>
Class Distribution
<br>
Walking: 424,400 (38.6%)
<br>
Jogging: 342,177 (31.2%)
<br>
Upstairs: 122,869 (11.2%)
<br>
Downstairs: 100,427 (9.1%)
<br>
Sitting: 59,939 (5.5%)
<br>
Standing: 48,395 (4.4%)
<br>
Contains all x,y,z  acceleration values 


<img src="images_readme\pie_chart.png"><br>



The reformatted dataset is formed by statistical measures taking 10 second window with 46 transformed attributes in it as given below-

XAVG, YAVG, ZAVG
XPEAK, YPEAK, ZPEAK 
XABSOLDEV, YABSOLDEV, ZABSOLDEV
XSTANDDEV, YSTANDDEV, ZSTANDDEV
RESULTANT is the average of the square roots of the sum of the values of each axis squared √(xi^2 + yi^2 + zi^2).
X0,X1X2…….Z8,Z9

Using the raw data we can plot the timeseries data showing the x,y,z acceleration of different users.For example While jogging the y acceleration would be maximum with less difference between the peaks.While standing the y accelration would be minimum.We can check the profiles for every activity.


<img src="images_readme/jogging_y_acceleration.png"><br>

The above figure shows the acceleration values for jogging.

<img src="images_readme/walking_yacceleration.png"><br>

The above figure shows the accelration values for walking.

Two techniques were used for sensor based HAR
1.Deep Neural Network(DNN)
2.Long short-term Memory(LSTM)


1.Deep neural network(DNN) was used on the reformated data
<br>
There are two models using DNN
<br>
Model 1 with 3 layers gives an accuracy of 87.4%
<br>

Model 2 with extra hidden layers gives an accuracy of 82%
<br>
<br>

2.LSTM model was used on raw time series data 
There are two models using the LSTM technique
<br>
Model 1 using 200 steps with Sliding window of 50 ((16692,200,3)) gives an accuracy of 86.1%
<br>

Model 2 using 200 steps with Non sliding window((4173,200,3)) gives an accuracy of 87.7%

<img src="images_readme/lstm_test.png"><br>
The above image shows the confusuin matrix for the best model (LSTM with no sliding window- 87.7% accuracy)






## Web App for sensor based HAR


<img src="images_readme/webpage.png"><br>

We designed a web app based on sensor based HAR.It gives the predicted class and original label on clicking the submit button




## Image-Based HAR
Transfer learning techniques were used to build an image classification model that recognizes various classes of human activity. 
The model was trained on images from the Stanford 40 dataset. You can find out more information about the data 
<a href="http://vision.stanford.edu/Datasets/40actions.html" target="_blank">here</a>.<br>
<img src="Image Classification\Output\plots\image.png"><br>
We used a only a portion of the full dataset, training on images from
the following ten action classes.<br>
### Classes
- Applauding
- Climbing
- Cooking
- Feeding a horse
- Holding an umbrella
- Jumping
- Playing guitar
- Riding a bike
- Riding a horse
- Walking the dog


The resulting model achieved a classification accuracy of 96%. 
