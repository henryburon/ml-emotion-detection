from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
np.set_printoptions(threshold=np.inf)  # Can use this to make it print out entire array
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import *
from sklearn.linear_model import LogisticRegression
import time

#####################################################################################################

# The main purpose of this file is to prepare and process the testing data
# In other words, it gets x_train, y_train, x_test, and y_test
# It then saves those values into specified .txt files, which can then be accessed by other python files
# You can get different variations of the training and testing data by modifying: 
#
# max_training_images_per_class
# max_testing_images_per_class
# num_orientations, num_cells, num_block
# and then also by commenting out certain classes (down from 7), if you only want to classify between say, angry and happy, or angry, happy, and neutral
#
# This file is also set up to run the Linear Regression model, but I recommend using it to get the data

#####################################################################################################

start_time = time.time()

#####################################################################################################
#####################################################################################################
# Load in the image data
#####################################################################################################
#####################################################################################################

# TRAINING data first
#######################

max_training_images_per_class = 3000

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
        angry_features_list,
        happy_features_list,
        neutral_features_list,
        sad_features_list,
        disgusted_features_list,
        fearful_features_list,
        surprised_features_list,

    ),
    axis=0,
)
y_train = (
    [0] * len(angry_features_list)
    + [1] * len(happy_features_list)
     + [2] * len(neutral_features_list)
    + [3] * len(sad_features_list)
    + [4] * len(disgusted_features_list)
    + [5] * len(fearful_features_list)
    + [6] * len(surprised_features_list)
)


# TESTING data second
#######################
x_test = np.concatenate(
    (
        angry_features_list_test,
        happy_features_list_test,
        neutral_features_list_test,
        sad_features_list_test,
        disgusted_features_list_test,
        fearful_features_list_test,
        surprised_features_list_test,
    ),
    axis=0,
)
y_test = (
    [0] * len(angry_features_list_test)
    + [1] * len(happy_features_list_test)
     + [2] * len(neutral_features_list_test)
    + [3] * len(sad_features_list_test)
    + [4] * len(disgusted_features_list_test)
    + [5] * len(fearful_features_list_test)
    + [6] * len(surprised_features_list_test)
)

#####################################################################

np.savetxt('x_train_3k.txt', x_train)
np.savetxt('y_train_3k.txt', y_train)
np.savetxt('x_test.txt', x_test)
np.savetxt('y_test.txt', y_test)


"""reg = 0.5
# train a logistic regression model on the training set
multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='ovr', max_iter=10000).fit(x_train, y_train)

# Make predictions
y_pred = multi_model.predict(x_test)

# Print report
print(classification_report(y_test, y_pred))

# weights = multi_model.coef_
# np.savetxt('7classes_logistic_regression_weights3.txt', weights)"""


##################################################################### TOTAL TIME

end_time = time.time()
total_time = end_time - start_time
print(f"Total time {total_time} seconds")