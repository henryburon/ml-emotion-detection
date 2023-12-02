from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import data, exposure
from skimage.feature import hog
from autograd.misc.flatten import flatten_func
from autograd import value_and_grad
np.set_printoptions(threshold=np.inf)  # Can use this to make it print out entire array
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from random import sample
from sklearn.multiclass import *
from sklearn.linear_model import LogisticRegression
import time
# from sklearn.externals import joblib
import joblib

# Start timer
start_time = time.time()

# Load data from the text files
x_train = np.loadtxt('training_data/x_train_3k.txt')
y_train = np.loadtxt('training_data/y_train_3k.txt')
x_test = np.loadtxt('x_test.txt')
y_test = np.loadtxt('y_test.txt')
# weights = np.loadtxt('7classes_logistic_regression_weights.txt') # Shape is (7, 1200).
# 7 sets of 1200 weights
# weights[0] = angry
# weights[1] = happy
# weights[2] = neutral
# weights[3] = sad
# weights[4] = disgusted
# weights[5] = fearful
# weights[6] = surprised





reg = 0.5
multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='ovr', max_iter=10000).fit(x_train, y_train)
# multi_model.coef_ = weights  # Set the coefficients manually


y_pred = multi_model.predict(x_test)

print(classification_report(y_test, y_pred))



joblib.dump(multi_model, '7_class_trained_logistic_regression_model.pk1')
# End timer and save weights
weights4 = multi_model.coef_
np.savetxt('7classes_logistic_regression_weights4.txt', weights4)
end_time = time.time()
total_time = end_time - start_time
print(f"Total time {total_time} seconds")