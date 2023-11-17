from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import data, exposure
from skimage.feature import hog
np.set_printoptions(threshold=np.inf) # Can use this to make it print out entire array


#####################################################################################################
# Load in the training data
#####################################################################################################

# Define the path to the parent folder containing subfolders for each class
train_parent_folder = 'data/train/'

# Define the class labels
class_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Loop through each class folder
for class_label in class_labels:
    folder_path = os.path.join(train_parent_folder, class_label)

    # Create a list to store the loaded images for the current class
    current_class_list = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image (you may want to add more sophisticated checks)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open the image using Pillow
            with Image.open(file_path) as img:
                img.load()
                current_class_list.append(img)

    # Append the list for the current class to the corresponding variable
    globals()["train_" + class_label] = current_class_list

# Each set of training data (distinct emotion) is loaded in as a corresponding list of images

#####################################################################################################
# Load in testing data
#####################################################################################################

# Define the path to the parent folder containing subfolders for each class
test_parent_folder = 'data/test/'

# Define the class labels
class_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Loop through each class folder
for class_label in class_labels:
    folder_path = os.path.join(test_parent_folder, class_label)

    # Create a list to store the loaded images for the current class
    current_class_list = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image (you may want to add more sophisticated checks)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open the image using Pillow
            with Image.open(file_path) as img:
                img.load()
                current_class_list.append(img)

    # Append the list for the current class to the corresponding variable
    globals()["test_" + class_label] = current_class_list

# Each set of testing data (distinct emotion) is loaded in as a corresponding list of images

#####################################################################################################
# Convert training images to numpy arrays, and normalize
#####################################################################################################

x_angry = [] # 3995 images. Each image is a 48x48 array. i.e. for each image, there are 48 arrays, each with 48 values.

for image in train_angry:
    image = np.array(image)/255 # Normalized
    x_angry.append(image)

x_happy = []

for image in train_happy:
    image = np.array(image)/255 # Normalized
    x_happy.append(image)

# There are 3995 images (arrays) in x_angry. Each of these arrays has 48x48 raw pixels/values (2304 total per image)
# It's possible I may have to make each image into one long array of 2304 features

#####################################################################################################
# Feature extraction: Apply HOG Feature Extraction/Image-Based Edge Histogram Feature, and Vectorize into 1-D format
#####################################################################################################

# The 'features_list' will be a list of 3995 arrays, each of X terms. Each array will be the features for an individual x

angry_features_list = []

for image_array in x_angry:
    features, hog_image = hog(image_array, orientations=8, pixels_per_cell=(12,12), cells_per_block=(1, 1), visualize=True)
    angry_features_list.append(features)

happy_features_list = []

for image_array in x_happy:
    features, hog_image = hog(image_array, orientations=8, pixels_per_cell=(12,12), cells_per_block=(1, 1), visualize=True)
    happy_features_list.append(features)

#####################################################################################################
# 
#####################################################################################################























"""
What to do next.

Instead of just using the raw pixels as my features (too many, doesn't take nearby stuff into account, etc.), I am going
to use the histogram oriented gradient (descent?) to extract features. Chapter 9.2 I think has some good information on thi
Basically, this uses edge detection to extract the features
I would like to do a manual implementation, 

Check this piazza link: https://piazza.com/class/ln1z3d64uzf1yw/post/66

May want to create a virtual environment if I'm installing all these packages, etc

"""

# image_np = x_angry[50]

# features, hog_image = hog(image_np, orientations=8, pixels_per_cell=(12,12), cells_per_block=(1, 1), visualize=True)
# # print(hog_image)
# # print(hog_image.shape)
# # print(len(hog_image))

# print("features:")
# print(features)
# print(features.shape)
# print(len(features))

# # Rescale histogram for better visualization
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# # print(hog_image_rescaled)
# # print(hog_image_rescaled.shape)
# # print(len(hog_image_rescaled))

# # Plot the original image and its corresponding HOG features
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# ax1.imshow(image_np, cmap=plt.cm.gray)
# ax1.set_title('Input Image')

# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('HOG Features')

# plt.show()



















#####################################################################################################
# Assign labels to training data (start with just the angry list)
#####################################################################################################

# coordinate_list = []

# # Y-value for an angry image is... 1
# for image in x_angry:
#     coordinate = (image, 1)
#     coordinate_list.append(coordinate)

# print(coordinate_list[44])



















#####################################################################################################
# Next
#####################################################################################################

"""
when i do numpy array, the first array is the top row of the image... 
closer to 0 represents black, closer to 255 represents white
"""


"""image = test_angry[7]
image.show()
print(image)

test_array = np.array(image)/255 # normalize

print(test_array)

print(test_array[0])
print(test_array[47])
"""







"""
what I'm going to need to do is extract the pixel/array values of each image
each of those will essentially be a feature, there will be 48x48.
that whole group of 48x48 represents an image, and that group of features, together, will be labeled.

start with just the images in the train_angry list.
so for each of these images, I'm going to need to get its pixel or numpy value or something
(remember to normalize to 1 by dividing by 255)
all these 48x48 pixels will be a total of 2304 features
i basically say those features correspond to an angry emotion

i then need to give a weight to each feature, so i can let my gradient descent do that... perceptron cost function tells me how good that is

"""











"""
angry: 1
disgusted: 2
fearful: 3
happy: 4
neutral: 5
sad: 6
surprised: 7
"""


# Going to use the raw pixel/array values as features
# sklearn preprocessing

# #####################################################################################################
# # Define model, perceptron, and gradient descent
# #####################################################################################################














# from PIL import Image
# import os

# # Define the path to the parent folder containing subfolders for each class
# train_parent_folder = '/home/henry/Desktop/Classes/EE_475/final_dataset/train/'

# # Define the class labels and their corresponding numeric values
# class_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
# class_numeric_values = {'angry': 1, 'disgusted': 2, 'fearful': 3, 'happy': 4, 'neutral': 5, 'sad': 6, 'surprised': 7}

# # Lists to store images and corresponding labels
# train_data = []
# train_labels = []

# # Loop through each class folder
# for class_label in class_labels:
#     folder_path = os.path.join(train_parent_folder, class_label)
#     label = class_numeric_values[class_label]

#     # Loop through each file in the folder
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)

#         # Check if the file is an image (you may want to add more sophisticated checks)
#         if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             # Open the image using Pillow
#             with Image.open(file_path) as img:
#                 img.load()
#                 train_data.append(img)
#                 train_labels.append(label)
#                 # turn image into array to analyze
# # Image data extractor
# # Now, train_data contains all the training images, and train_labels contains their corresponding labels
