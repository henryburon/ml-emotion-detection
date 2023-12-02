
from PIL import Image
import os
import numpy as np
import time
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
np.set_printoptions(threshold=np.inf)  # Can use this to make it print out entire array
import joblib


start_time = time.time()

weights = np.loadtxt('weights/7classes_logistic_regression_weights.txt')
loaded_model = joblib.load('trained_model/7_class_trained_logistic_regression_model.pk1')

#####################################################################################################
# Load in images, reshape to 48x48 pixels, and make grayscale
#####################################################################################################

folder_path = 'images/unknown/original'  # Replace this with the path to your folder containing images

image_list = []

# Iterate through all images in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        try:
            with Image.open(file_path) as img:
                # Process the image here (e.g., display or perform operations)
                # img.show()  # Opens the image using the default viewer
                img.load()
                img = img.resize((48, 48)).convert('L')
                image_list.append(img)
        except Exception as e:
            print(f"Could not open {filename}: {e}")

#####################################################################################################
# Convert to 48x48 numerical arrays, normalize, extract hog features
#####################################################################################################

image_array_list = []

for image in image_list:
    image = np.array(image)/255
    image_array_list.append(image)

image_features_list = []
image_hog_list = []

num_orientation = 12
num_cells = 8
num_block = 2

for image_array in image_array_list:
    features, hog_image = hog(
        image_array,
        orientations=num_orientation,
        pixels_per_cell=(num_cells, num_cells),
        cells_per_block=(num_block, num_block),
        visualize=True,
    )
    image_features_list.append(features)
    image_hog_list.append(hog_image)

#####################################################################################################
# Run Linear Regression
#####################################################################################################


# reg = 0.5
# train a logistic regression model on the training set
# multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='ovr', max_iter=10000).fit(x_train, y_train)

# multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='ovr', max_iter=10000)

# multi_model.coef_ = weights

# predictions = multi_model.predict(image_features_list)

predictions = loaded_model.predict(image_features_list)
print(predictions)

# Shape is (7, 1200).
# 7 sets of 1200 weights
# weights[0] = angry
# weights[1] = happy
# weights[2] = neutral
# weights[3] = sad
# weights[4] = disgusted
# weights[5] = fearful
# weights[6] = surprised






end_time = time.time()
total_time = end_time - start_time
print(f"Total time {total_time} seconds")
