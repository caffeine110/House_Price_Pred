
## AIM	: Predictive analysis of House Price using House Features in a locality using Machine Learning.
	: ( House price prediction.)
### Author	: predict11

### Introduction
It very Hard to predict the price of the house for machine
problem can be solved using the Neural Network


### Keywords 
Keywords : Machine Learning, Data Analysis, Satastics, DNN, Numpy, Pandas.

## Tools
PreRequirements :

		 LIBRARIES	: Pandas,Numpy, Sklearn, Keras, TensorFlow, matplotlib, csv.
		 IDE		: spyder

###
Abstraact	: We have studied the data manipulation libraries such as Numpy and Pandas for handling the huge dataset of House Data.
		  using matplot library we can visualise all the implimented modules.
		  Using sk-learn we can import test-train split method which divides the whole data into test and train cases.
		  Using keras we can build the DNN model with Sequential layers.
		  TensorFlow is an alternative library which allows to create ML model using Estemators and Tensors.




# procedure to run
Procedure : 

	1). Exctraction :
		Dataset is exctrated from kaggle
	2). Preporcessing
		Run the preprocessing.py file to preprocess the downloaded data.
	3). Model Training
		Run the models.py file to fit data to model.
		While the model if trained program is under exicution and after complition apply the prediction steps.
	4). prediction
		To predict the best price of house put the data tupule in test Pred_test cases.
		Output is shown in single floating point Number as a Price of house


# Evaluation Plan

We solved problems at occured at each step to improve accuracy.

As this is a Regression problem it is difficult to measure performance of Regressor  model than the Classification One.

So we have mesure the performance of model using :
	Varience score, Mean Square Error, Mean Absolute Error, Median Absolute Error and by plotting Graphs...


### key Metrics :
Variance—
	In terms of linear regression, variance Is a measure of how far observed
	values differ from the average of predicted values.

	Varience Score is :  0.88
	Idely it is 1

Mean Square Error (MSE)—
	It is the average of the square of the errors.
	The larger the number the larger the error.

	Mean Square Error :  15785156063.40

Absolute errorse(AE)—
	It is a difference between two continues variables

	Mean Absolute Error :  69836.26
	Median Absulute Error :  40913.34

This are preety large Numbers because price of house are alwayes in lacs and crores


# Optimisation :

### Parameter Tuning
We have tued the Parameters from the 

	Train-Test split from 60-40 ... 80-20 and get the best accuracy at 80-20

	Varing from 1...5 we got best accuracy at Dense Layers : 3

	Number of Neurons in Layer each layers :
	We got best accuracy at layers at :
		: 86 Neurons at input layer as Parameters are 86
		: 64 Neurons at First Hidden layer
		: 32 Neurons at Second Hidden layer
		: 1 Neuron at output layer as there is only 1 Output Price

	Tuned No of epoches 100 to 3000 and found best accuracy at 500 epochs
	Batch size 16
