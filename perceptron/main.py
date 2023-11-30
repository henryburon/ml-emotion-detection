from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import data, exposure
from skimage.feature import hog
from autograd.misc.flatten import flatten_func
from autograd import value_and_grad
np.set_printoptions(threshold=np.inf) # Can use this to make it print out entire array

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


max_images_per_class = 3900 # added this. I think 3500 images can get up to 61% accuracy

number = max_images_per_class # just using this as a variable for creating the y labeling lists

#####################################################################################################
# Load in the training data
#####################################################################################################

train_parent_folder = 'data/train/'

# Define the class labels
class_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Loop through each class folder
for class_label in class_labels:

    image_count = 0 # added this

    folder_path = os.path.join(train_parent_folder, class_label)

    # Create a list to store the loaded images for the current class
    current_class_list = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if image_count >= max_images_per_class: # added this
            break
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image (you may want to add more sophisticated checks)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open the image using Pillow
            with Image.open(file_path) as img:
                img.load()
                current_class_list.append(img)
                image_count += 1 # added this

    # Append the list for the current class to the corresponding variable
    globals()["train_" + class_label] = current_class_list

# Each set of training data (distinct emotion) is loaded in as a corresponding list of images

#####################################################################################################
# Load in testing data
#####################################################################################################

test_parent_folder = 'data/test/'

# Define the class labels
class_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Loop through each class folder
for class_label in class_labels:

    image_count = 0 # added this

    folder_path = os.path.join(test_parent_folder, class_label)

    # Create a list to store the loaded images for the current class
    current_class_list = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if image_count >= 200: # added this
            break
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image (you may want to add more sophisticated checks)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open the image using Pillow
            with Image.open(file_path) as img:
                img.load()
                current_class_list.append(img)
                image_count += 1

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

x_neutral = []

for image in train_neutral:
    image = np.array(image)/255 # Normalized
    x_neutral.append(image)

# There are 3995 images (arrays) in x_angry. Each of these arrays has 48x48 raw pixels/values (2304 total per image)

#####################################################################################################
# Convert testing images to numpy arrays, and normalize
#####################################################################################################

x_angry2 = [] # 3995 images. Each image is a 48x48 array. i.e. for each image, there are 48 arrays, each with 48 values.

for image in test_angry:
    image = np.array(image)/255 # Normalized
    x_angry2.append(image)

x_happy2 = []

for image in test_happy:
    image = np.array(image)/255 # Normalized
    x_happy2.append(image)

x_neutral2 = []

for image in test_neutral:
    image = np.array(image)/255 # Normalized
    x_neutral2.append(image)

# There are 3995 images (arrays) in x_angry. Each of these arrays has 48x48 raw pixels/values (2304 total per image)

#####################################################################################################
# Feature extraction: Apply HOG Feature Extraction/Image-Based Edge Histogram Feature, and Vectorize into 1-D format (training)
#####################################################################################################

# The 'features_list' will be a list of 3995 arrays, each of X terms. Each array will be the features for an individual image.

# Orientations: increasing # allows for more finely quantized gradient directions within each cell. Higher values captures more detail.
# Pixels per cell: Larger values result in larger spatial areas covered by each cell. Needs to be a factor of 48. Smaller values = more detail. Probably want 8, 12, or 16.
# Cells per block: 

##########


orientations_num = 12
cells = 8
block = 2



angry_features_list = []
angry_hog_image_list = []

for image_array in x_angry:
    features, hog_image = hog(image_array, orientations=orientations_num, pixels_per_cell=(cells, cells), cells_per_block=(block, block), visualize=True)
    angry_features_list.append(features)
    angry_hog_image_list.append(hog_image)

##########

happy_features_list = []
happy_hog_image_list = []

for image_array in x_happy:
    features, hog_image = hog(image_array, orientations=orientations_num, pixels_per_cell=(cells, cells), cells_per_block=(block, block), visualize=True)
    happy_features_list.append(features)
    happy_hog_image_list.append(hog_image)

##########

neutral_features_list = []
neutral_hog_image_list = []

for image_array in x_neutral:
    features, hog_image = hog(image_array, orientations=orientations_num, pixels_per_cell=(cells, cells), cells_per_block=(block, block), visualize=True)
    neutral_features_list.append(features)
    neutral_hog_image_list.append(hog_image)

# The features list contains the extracted features.
# The hog image list contains the visual representation of the HOG features

#####################################################################################################
# Feature extraction: Apply HOG Feature Extraction/Image-Based Edge Histogram Feature, and Vectorize into 1-D format (testing)
#####################################################################################################

# The 'features_list' will be a list of 3995 arrays, each of X terms. Each array will be the features for an individual image.

# Orientations: increasing # allows for more finely quantized gradient directions within each cell. Higher values captures more detail.
# Pixels per cell: Larger values result in larger spatial areas covered by each cell. Needs to be a factor of 48. Smaller values = more detail. Probably want 8, 12, or 16.
# Cells per block: 

##########

angry_features_list2 = []
angry_hog_image_list2 = []


for image_array in x_angry2:
    features, hog_image = hog(image_array, orientations=orientations_num, pixels_per_cell=(cells, cells), cells_per_block=(block, block), visualize=True)
    angry_features_list2.append(features)
    angry_hog_image_list2.append(hog_image)

##########

happy_features_list2 = []
happy_hog_image_list2 = []

for image_array in x_happy2:
    features, hog_image = hog(image_array, orientations=orientations_num, pixels_per_cell=(cells, cells), cells_per_block=(block, block), visualize=True)
    happy_features_list2.append(features)
    happy_hog_image_list2.append(hog_image)

##########

neutral_features_list2 = []
neutral_hog_image_list2 = []

for image_array in x_neutral2:
    features, hog_image = hog(image_array, orientations=orientations_num, pixels_per_cell=(cells, cells), cells_per_block=(block, block), visualize=True)
    neutral_features_list2.append(features)
    neutral_hog_image_list2.append(hog_image)

# The features list contains the extracted features.
# The hog image list contains the visual representation of the HOG features

#####################################################################################################
# Example for showing HOG features next to image.
#####################################################################################################

"""desired_image = 64

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(x_happy[desired_image], cmap='gray')
plt.title(f'Original Image {desired_image}')

plt.subplot(1, 2, 2)
plt.imshow(happy_hog_image_list[desired_image], cmap='gray')
plt.title(f'HOG Features for Image {desired_image}')
plt.show()"""

#####################################################################################################
# The goal with the manual implementation would be to train weights corresponding to each feature. Those weights are used in the model (linear combination) and
# added up to give a value, which is then associated with a certain label.
# The key for ML is to train a list of weights
# The weights are in an ordered array, which, when applied to their corresponding feature, add up with the model to give a value
# so the result from training an ML model is really just that list of weights
#####################################################################################################

#####################################################################################################
# Start with 2-class classification
#####################################################################################################

# There are 300 arrays in happy_features_list. Each array has 288 features (when pixels_per_cell is (8,8))
# It's a list of arrays



# Create training data

happy_features_list = np.array(happy_features_list) # Shape is (300, 128), so have 128 features for each of the 300 images
angry_features_list = np.array(angry_features_list)
neutral_features_list = np.array(neutral_features_list)

x_train = np.concatenate((happy_features_list,angry_features_list,neutral_features_list), axis=0)
y_train = [1]*number + [2]*number + [3]*number



# Create testing data

happy_features_list2 = np.array(happy_features_list2) # Shape is (300, 128), so have 128 features for each of the 300 images
angry_features_list2 = np.array(angry_features_list2)
neutral_features_list2 = np.array(neutral_features_list2)

x_test = np.concatenate((happy_features_list2,angry_features_list2,neutral_features_list2), axis=0)
y_test = [1]*200 + [2]*200 + [3]*200



perceptron = Perceptron(max_iter=1000, random_state=42)
classifier = OneVsRestClassifier(perceptron)
classifier.fit(x_train,y_train)


# # # svc_model = SVC(kernel='linear', C=1.0)  # Example of SVC with a linear kernel and C=1.0

# # # # Fit the model on your training data
# # # svc_model.fit(x_train, y_train)

y_pred = classifier.predict(x_test)



# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)






# Access the learned weights
weights = classifier.estimators_[0].coef_  # Get weights for the first class


print(len(weights[0])) # This is a list of the weights for each of the features. I think perceptron is the model that adds them all up? idk
print(len(happy_features_list[0]))