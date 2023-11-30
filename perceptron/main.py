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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


max_images_per_class = 1200 # added this. I think 3000 images can get up to 61% accuracy

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



# perceptron = Perceptron(max_iter=1000, random_state=42)
# clf = OneVsRestClassifier(perceptron)
# clf.fit(x_train,y_train)


# # svc_model = SVC(kernel='linear', C=1.0)  # Example of SVC with a linear kernel and C=1.0

# # # Fit the model on your training data
# # svc_model.fit(x_train, y_train)

# y_pred = clf.predict(x_test)

# # Evaluate accuracy on test data

# accuracy = accuracy_score(y_test, y_pred) # y_test is the true y value...
# print(f"Accuracy: {accuracy}")

svm_model = SVC()

# Define a grid of hyperparameters to search
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel type
    'gamma': ['scale', 'auto']  # Kernel coefficient (only for 'rbf' kernel)
}

# Perform a grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Get the best model found by the grid search
best_svm_model = grid_search.best_estimator_

# Train the best model on the entire training set
best_svm_model.fit(x_train, y_train)

# Predict using the trained model
y_pred_svm = best_svm_model.predict(x_test)

# Evaluate accuracy on test data
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy using SVM: {accuracy_svm}")











































"""# Initialize the multiclass Perceptron
perceptron = Perceptron(max_iter=1000, random_state=42)  # You can adjust max_iter as needed

# Train the Perceptron using your training data
perceptron.fit(x_train2, y_train2)

# Predict using the trained Perceptron on test data
y_pred = perceptron.predict(x_test2)

# Evaluate accuracy on test data
accuracy = accuracy_score(y_test2, y_pred)
print(f"Accuracy: {accuracy}")"""


"""
# I believe orientations refers to kernels. More orientations would capture finer detail, but make computations more intense. 8 is good.
# Pixels per cell determines the size of each cell (window?) over which the HOG is computed. Smaller number is finer detail.
# Might want to make pixels per cell a bit smaller... like (6,6) or (4,4)

#####################################################################################################
# Assign labels to training data (start with just the angry and happy lists)
#####################################################################################################

# Combine the lists of features, and then make corresponding lists of labels

# Assign "0" to angry, "1" to happy

# x = np.array(angry_features_list +  happy_features_list)
# y = np.array([0] * len(angry_features_list) + [1] * len(happy_features_list))

x = np.array(angry_features_list)
y = np.array([0] * len(angry_features_list))

print(x.shape)
print(y.shape)

#####################################################################################################
# Define multi-class perceptron and other necessary functions
#####################################################################################################

# compute C linear combinations of input point, one per classifier
def model(x,w):
    a = w[0] + np.dot(x.T,w[1:])
    return a.T

lam = 10**-5  # our regularization paramter 
def multiclass_perceptron(w):        
    # pre-compute predictions on all points
    all_evals = model(x,w)
    
    # compute maximum across data points
    a = np.max(all_evals,axis = 0)    

    # compute cost in compact form using numpy broadcasting
    b = all_evals[y.astype(int).flatten(),np.arange(np.size(y))]
    cost = np.sum(a - b)
    
    # add regularizer
    cost = cost + lam*np.linalg.norm(w[1:,:],'fro')**2
    
    # return average
    return cost/float(np.size(y))

def gradient_descent(g,alpha_choice,max_its,w):
    # flatten the input function to more easily deal with costs that have layers of parameters
    g_flat, unflatten, w = flatten_func(g, w) # note here the output 'w' is also flattened

    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice

        # evaluate the gradient, store current (unflattened) weights and cost function value
        cost_eval,grad_eval = gradient(w)
        weight_history.append(unflatten(w))
        cost_history.append(cost_eval)

        # take gradient descent step
        w = w - alpha*grad_eval

    # collect final weights
    weight_history.append(unflatten(w))
    # compute final cost function value via g itself (since we aren't computing
    # the gradient at the final step we don't get the final cost function value
    # via the Automatic Differentiatoor)
    cost_history.append(g_flat(w))
    return weight_history,cost_history

#####################################################################################################
# Run
#####################################################################################################

# Assume num_features is the number of features extracted using HOG
# Assume num_classes is the total number of classes
num_features = len(angry_features_list[0])  # Assuming the length of the first feature set. same number of features for everything
num_classes = 2  # Assuming two classes (angry and happy)

initial_weights = np.zeros((num_features + 1, num_classes))


# print(num_features)
# print(initial_weights.shape)



max_iterations = 1000  # Set the maximum number of iterations
learned_weights, cost_history = gradient_descent(multiclass_perceptron, 0.01, max_iterations, initial_weights)




"""
# Having some issues with the dimensions...

# Should probably go back and check that every array so far is what I want it to be...
# Make sure I have the correct gradient descent
# Go check the 9.2 on mac, a clue




"""




# g = multiclass_perceptron; w = 0.1*np.random.randn(3,3); max_its = 1000; alpha_choice = 10**(-1);
# weight_history,cost_history = gradient_descent(g,alpha_choice,max_its,w)






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











"""
































