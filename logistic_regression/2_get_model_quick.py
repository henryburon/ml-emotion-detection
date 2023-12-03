import numpy as np
np.set_printoptions(threshold=np.inf)  # Can use this to make it print out entire array
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import *
from sklearn.linear_model import LogisticRegression
import time
import joblib


##########################################################################################

# The purpose of this file is to get (and save) the linear regression model quickly as a .pk1 file (i.e. without having to prepare/process the training and testing data)
# You don't have to prepare/process the training or testing data, just load it in (there are different options based on training sample size)
# Simply load the desired data in, and run the file. It will save a linear regression model which can be used to quickly implement the classification
 
##########################################################################################

# weights = np.loadtxt('7classes_logistic_regression_weights.txt') # Shape is (7, 1200).
# 7 sets of 1200 weights
# weights[0] = angry
# weights[1] = happy
# weights[2] = neutral
# weights[3] = sad
# weights[4] = disgusted
# weights[5] = fearful
# weights[6] = surprised

# Start timer
start_time = time.time()

# Load in data from the .txt files
# Each training data might be based on a different amount of samples and classes
x_train = np.loadtxt('loaded_data/x_happy_neutral_surprised.txt')
y_train = np.loadtxt('loaded_data/y_happy_neutral_surprised.txt')
x_test = np.loadtxt('loaded_data/x_test_1k.txt')
y_test = np.loadtxt('loaded_data/y_test_1k.txt')

# Run the linear regression model
reg = 0.5
multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='ovr', max_iter=10000).fit(x_train, y_train)
# multi_model.coef_ = weights  # Set the coefficients manually

# Make predictions
y_pred = multi_model.predict(x_test)

# Get classification report
print(classification_report(y_test, y_pred))

### Save the linear regression model ###
joblib.dump(multi_model, 'logistic_regression/trained_model/happy_neutral_surprised.pk1')

# End timer
# weights4 = multi_model.coef_
end_time = time.time()
total_time = end_time - start_time
print(f"Total time {total_time} seconds")




