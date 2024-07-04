# Precision_Agriculture
The steps involved in this system implementation are :-

a)Acquisition of Training Dataset: 
The accuracy of any machine learning algorithm depends on the amount of parameters and therefore correctness of the training dataset. For the system, we are using various datasets all downloaded from government website and kaggle.
Datasets include:-
Yield dataset, Fertilizer dataset, Soil nutrient content dataset, Rainfall Temperature dataset

b) Data Preprocessing: This step includes replacing the null and 0 values for yield by -1 so that it does not effect the overall prediction. Further we had to encode the dataset so that it could be fed into the our ML models.

c) Training ML model: After the preprocessing step we used the dataset to train different machine learning models like Random forest, Decision Tree, Support Vector Machine(SVM) and Logistic regression to attain accuracy as high as possible.

d) Model Evaluation and Saving Model:  All the ML models which are trained would be evaluated by comparing their performance (Evaluations Metrics) and Final efficient model is saved using pickle library.

e) Model Exportation and Integration with Web app: The saved efficient ML model would be integrated with Flask Web Application which would further meant for prediction in user friendly web interface.

f) Real-time Testing of Application: This step includes real-time testing of our whole application using an IOT system which consists of
	a).Soil NPK Sensor,
	b).Capacitive Soil Moisture Sensor,
	c).Temperature Sensor,
	d).Wireless Transceiver module and
	e). Arduino Nano board.
 
Soil NPK sensor, Soil Moisture and Temperature sensors are dipped into soil along with help of Arduino Nano board to acquire all the features of soil. We get real-time data of soil like N, P, K, Moisture, Temperature, etc which are used to test our pre-built Web Application manually and obtain the predictions done.
