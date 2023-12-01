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

start_time = time.time()

#####################################################################################################
#####################################################################################################
# Load in the image data
#####################################################################################################
#####################################################################################################

# TRAINING data first
#######################

max_training_images_per_class = 4000

train_parent_folder = "images/train/"
class_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Loop through each class folder
for class_label in class_labels:
    image_count = 0

    folder_path = os.path.join(train_parent_folder, class_label)

    # Create a list to store the loaded images for the current class
    current_class_list = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if image_count >= max_training_images_per_class:  # added this
            break
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image (you may want to add more sophisticated checks)
        if os.path.isfile(file_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            # Open the image using Pillow
            with Image.open(file_path) as img:
                img.load()
                current_class_list.append(img)
                image_count += 1

    # Append the list for the current class to the corresponding variable
    globals()["train_" + class_label] = current_class_list

# TESTING data second
#######################

max_testing_images_per_class = 1000

test_parent_folder = "images/test/"
class_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Loop through each class folder
for class_label in class_labels:
    image_count = 0  # added this

    folder_path = os.path.join(test_parent_folder, class_label)

    # Create a list to store the loaded images for the current class
    current_class_list = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if image_count >= max_testing_images_per_class:
            break
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image (you may want to add more sophisticated checks)
        if os.path.isfile(file_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            # Open the image using Pillow
            with Image.open(file_path) as img:
                img.load()
                current_class_list.append(img)
                image_count += 1

    # Append the list for the current class to the corresponding variable
    globals()["test_" + class_label] = current_class_list

#####################################################################################################
#####################################################################################################
# Convert to arrays and normalize
#####################################################################################################
#####################################################################################################

# TRAINING data first
#######################

x_angry = (
    []
)  # 3995 images. Each image is a 48x48 array. i.e. for each image, there are 48 arrays, each with 48 values.

for image in train_angry:
    image = np.array(image) / 255
    x_angry.append(image)

x_disgusted = []

for image in train_disgusted:
    image = np.array(image) / 255
    x_disgusted.append(image)

x_fearful = []

for image in train_fearful:
    image = np.array(image) / 255
    x_fearful.append(image)

x_happy = []

for image in train_happy:
    image = np.array(image) / 255
    x_happy.append(image)

x_neutral = []

for image in train_neutral:
    image = np.array(image)/255
    x_neutral.append(image)

x_sad = []

for image in train_sad:
    image = np.array(image)/255
    x_sad.append(image)

x_surprised = []

for image in train_surprised:
    image = np.array(image)/255
    x_surprised.append(image)

# TESTING data second
#######################

x_angry_test = []

for image in test_angry:
    image = np.array(image) / 255
    x_angry_test.append(image)

x_disgusted_test = []

for image in test_disgusted:
    image = np.array(image) / 255
    x_disgusted_test.append(image)

x_fearful_test = []

for image in test_fearful:
    image = np.array(image) / 255
    x_fearful_test.append(image)

x_happy_test = []

for image in test_happy:
    image = np.array(image) / 255
    x_happy_test.append(image)

x_neutral_test = []

for image in test_neutral:
    image = np.array(image)/255
    x_neutral_test.append(image)

x_sad_test = []

for image in test_sad:
    image = np.array(image)/255
    x_sad_test.append(image)

x_surprised_test = []

for image in test_surprised:
    image = np.array(image)/255
    x_surprised_test.append(image)

#####################################################################################################
#####################################################################################################
# Feature extraction: Apply HOG Feature Extraction/Image-Based Edge Histogram Feature, and Vectorize into 1-D format (training)
#####################################################################################################
#####################################################################################################

num_orientation = 12
num_cells = 8
num_block = 2

# TRAINING data first
#######################

angry_features_list = []
angry_hog_image_list = []

for image_array in x_angry:
    features, hog_image = hog(
        image_array,
        orientations=num_orientation,
        pixels_per_cell=(num_cells, num_cells),
        cells_per_block=(num_block, num_block),
        visualize=True,
    )
    angry_features_list.append(features)
    angry_hog_image_list.append(hog_image)

disgusted_features_list = []
disgusted_hog_image_list = []

for image_array in x_disgusted:
    features, hog_image = hog(
        image_array,
        orientations=num_orientation,
        pixels_per_cell=(num_cells, num_cells),
        cells_per_block=(num_block, num_block),
        visualize=True,
    )
    disgusted_features_list.append(features)
    disgusted_hog_image_list.append(hog_image)

fearful_features_list = []
fearful_hog_image_list = []

for image_array in x_fearful:
    features, hog_image = hog(
        image_array,
        orientations=num_orientation,
        pixels_per_cell=(num_cells, num_cells),
        cells_per_block=(num_block, num_block),
        visualize=True,
    )
    fearful_features_list.append(features)
    fearful_hog_image_list.append(hog_image)

happy_features_list = []
happy_hog_image_list = []

for image_array in x_happy:
    features, hog_image = hog(
        image_array,
        orientations=num_orientation,
        pixels_per_cell=(num_cells, num_cells),
        cells_per_block=(num_block, num_block),
        visualize=True,
    )
    happy_features_list.append(features)
    happy_hog_image_list.append(hog_image)

neutral_features_list = []
neutral_hog_image_list = []

for image_array in x_neutral:
    features, hog_image = hog(image_array, orientations=num_orientation, pixels_per_cell=(num_cells, num_cells), cells_per_block=(num_block, num_block), visualize=True)
    neutral_features_list.append(features)
    neutral_hog_image_list.append(hog_image)

sad_features_list = []
sad_hog_image_list = []

for image_array in x_sad:
    features, hog_image = hog(image_array, orientations=num_orientation, pixels_per_cell=(num_cells, num_cells), cells_per_block=(num_block, num_block), visualize=True)
    sad_features_list.append(features)
    sad_hog_image_list.append(hog_image)

surprised_features_list = []
surprised_hog_image_list = []

for image_array in x_surprised:
    features, hog_image = hog(image_array, orientations=num_orientation, pixels_per_cell=(num_cells, num_cells), cells_per_block=(num_block, num_block), visualize=True)
    surprised_features_list.append(features)
    surprised_hog_image_list.append(hog_image)

# TESTING data second
#######################

angry_features_list_test = []
angry_hog_image_list_test = []

for image_array in x_angry_test:
    features, hog_image = hog(
        image_array,
        orientations=num_orientation,
        pixels_per_cell=(num_cells, num_cells),
        cells_per_block=(num_block, num_block),
        visualize=True,
    )
    angry_features_list_test.append(features)
    angry_hog_image_list_test.append(hog_image)

disgusted_features_list_test = []
disgusted_hog_image_list_test = []

for image_array in x_disgusted_test:
    features, hog_image = hog(
        image_array,
        orientations=num_orientation,
        pixels_per_cell=(num_cells, num_cells),
        cells_per_block=(num_block, num_block),
        visualize=True,
    )
    disgusted_features_list_test.append(features)
    disgusted_hog_image_list_test.append(hog_image)

fearful_features_list_test = []
fearful_hog_image_list_test = []

for image_array in x_fearful_test:
    features, hog_image = hog(
        image_array,
        orientations=num_orientation,
        pixels_per_cell=(num_cells, num_cells),
        cells_per_block=(num_block, num_block),
        visualize=True,
    )
    fearful_features_list_test.append(features)
    fearful_hog_image_list_test.append(hog_image)

happy_features_list_test = []
happy_hog_image_list_test = []

for image_array in x_happy_test:
    features, hog_image = hog(
        image_array,
        orientations=num_orientation,
        pixels_per_cell=(num_cells, num_cells),
        cells_per_block=(num_block, num_block),
        visualize=True,
    )
    happy_features_list_test.append(features)
    happy_hog_image_list_test.append(hog_image)

neutral_features_list_test = []
neutral_hog_image_list_test = []

for image_array in x_neutral_test:
    features, hog_image = hog(image_array, orientations=num_orientation, pixels_per_cell=(num_cells, num_cells), cells_per_block=(num_block, num_block), visualize=True)
    neutral_features_list_test.append(features)
    neutral_hog_image_list_test.append(hog_image)

sad_features_list_test = []
sad_hog_image_list_test = []

for image_array in x_sad_test:
    features, hog_image = hog(image_array, orientations=num_orientation, pixels_per_cell=(num_cells, num_cells), cells_per_block=(num_block, num_block), visualize=True)
    sad_features_list_test.append(features)
    sad_hog_image_list_test.append(hog_image)

surprised_features_list_test = []
surprised_hog_image_list_test = []

for image_array in x_surprised_test:
    features, hog_image = hog(image_array, orientations=num_orientation, pixels_per_cell=(num_cells, num_cells), cells_per_block=(num_block, num_block), visualize=True)
    surprised_features_list_test.append(features)
    surprised_hog_image_list_test.append(hog_image)



# sample = np.array(surprised_features_list)
# print(f"Shape of data: {sample.shape}")
# for above, it's (# of data samples, # of features), such as (500, 1200)

#####################################################################################################
# Example for showing HOG features next to image.
#####################################################################################################
"""
desired_image = 64

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(x_happy_test[desired_image], cmap='gray')
plt.title(f'Original Image {desired_image}')

plt.subplot(1, 2, 2)
plt.imshow(happy_hog_image_list_test[desired_image], cmap='gray')
plt.title(f'HOG Features for Image {desired_image}')
plt.show()"""

#####################################################################################################
#####################################################################################################
# Classification
#####################################################################################################
#####################################################################################################

# TRAINING data first
#######################
x_train = np.concatenate(
    (
        # angry_features_list,
        # happy_features_list,
        neutral_features_list,
        # sad_features_list,
        # disgusted_features_list,
        # fearful_features_list,
        surprised_features_list,

    ),
    axis=0,
)
y_train = (
    # [0] * len(angry_features_list)
    # + [1] * len(happy_features_list)
     [2] * len(neutral_features_list)
    # + [3] * len(sad_features_list)
    # + [4] * len(disgusted_features_list)
    # + [5] * len(fearful_features_list)
    + [6] * len(surprised_features_list)
)


# TESTING data second
#######################
x_test = np.concatenate(
    (
        # angry_features_list_test,
        # happy_features_list_test,
        neutral_features_list_test,
        # sad_features_list_test,
        # disgusted_features_list_test,
        # fearful_features_list_test,
        surprised_features_list_test,
    ),
    axis=0,
)
y_test = (
    # [0] * len(angry_features_list_test)
    # + [1] * len(happy_features_list_test)
     [2] * len(neutral_features_list_test)
    # + [3] * len(sad_features_list_test)
    # + [4] * len(disgusted_features_list_test)
    # + [5] * len(fearful_features_list_test)
    + [6] * len(surprised_features_list_test)
)

# Now, run the algorithm

# perceptron = Perceptron(max_iter=1000, random_state=35)
# classifier = OneVsRestClassifier(perceptron)
# classifier.fit(x_train,y_train)

# y_pred = classifier.predict(x_test)
# # weights = classifier.estimators_[0].coef_

# # # This is the 1200 weights (1 for each feature)
# # print("Shape of Learned Weights:")

# # print(weights.shape)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# class_report = classification_report(y_test, y_pred)
# print("Classification Report:\n", class_report)

##################################################################### GRID SEARCH PERCEPTRON

"""# Define the parameter grid to search through
param_grid_perceptron = {
    'max_iter': [100],  # Define different values to test
    'tol': [1e-3],  # Change tolerance values
    # Add more hyperparameters as needed
}

# Create the Perceptron classifier
perceptron = Perceptron()

# Perform Grid Search
grid_search_perceptron = GridSearchCV(perceptron, param_grid=param_grid_perceptron, cv=5)
grid_search_perceptron.fit(x_train, y_train)

# Print the best parameters found
print("Best Parameters for Perceptron:")
print(grid_search_perceptron.best_params_)

# Get the best model
best_perceptron = grid_search_perceptron.best_estimator_

# Evaluate the best model on the test set
y_pred_perceptron = best_perceptron.predict(x_test)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
print(f"Accuracy of Perceptron after Grid Search: {accuracy_perceptron:.2f}")

class_report = classification_report(y_test, y_pred_perceptron)
print("Classification Report:\n", class_report)"""

##################################################################### WEIGHTS AND INTERCEPT

# # After fitting the model
# weights = best_perceptron.coef_
# intercept = best_perceptron.intercept_

# Save weights to a text file
# with open('perceptron_weights.csv', 'w') as file:
#     for weight in weights:
#         file.write(', '.join(str(w) for w in weight) + '\n')

# # Save intercept to a text file
# with open('perceptron_intercept.txt', 'w') as file:
#     file.write(', '.join(str(i) for i in intercept))

##################################################################### SOFTMAX



# # Create an MLPClassifier (Softmax classifier)
# softmax_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam', max_iter=500, random_state=2)

# # Fit the classifier to the training data
# softmax_classifier.fit(x_train, y_train)

# # Predict using the trained classifier
# y_pred_softmax = softmax_classifier.predict(x_test)

# # Get the weights of the trained MLPClassifier
# weights_softmax = softmax_classifier.coefs_

# # Print the shape of the learned weights
# for i, weight_matrix in enumerate(weights_softmax):
#     print(f"Shape of Weights for Layer {i+1}: {weight_matrix.shape}")

# # Calculate accuracy and display classification report
# accuracy_softmax = accuracy_score(y_test, y_pred_softmax)
# print(f"Accuracy of Softmax Classifier: {accuracy_softmax:.2f}")

# class_report_softmax = classification_report(y_test, y_pred_softmax)
# print("Classification Report (Softmax Classifier):\n", class_report_softmax)


##################################################################### TOTAL TIME






reg = 0.1
# train a logistic regression model on the training set
multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000).fit(x_train, y_train)
print (multi_model)

y_pred = multi_model.predict(x_test)


print(classification_report(y_test, y_pred))









end_time = time.time()
total_time = end_time - start_time
print(f"Total time {total_time} seconds")

