import os
from skimage.io import imread
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Prepare Data
dataset_path_train = "/home/kashedd/Desktop/EE _475/FinalProject/archive/train"
dataset_path_test = "/home/kashedd/Desktop/EE _475/FinalProject/archive/test"
emotion_classes = os.listdir(dataset_path_train)

train_images = []
train_labels = []

test_images = []
test_labels = []

for i, emotion_class in enumerate(emotion_classes):
    train_emotion_path = os.path.join(dataset_path_train, emotion_class)
    test_emotion_path = os.path.join(dataset_path_test, emotion_class)

    for filename in os.listdir(train_emotion_path):
        if filename.endswith('png'):
            img_path = os.path.join(train_emotion_path, filename)
            img = imread(img_path)
            train_images.append(img)
            train_labels.append(i)
            

    for filename in os.listdir(test_emotion_path):
        if filename.endswith('png'):
            img_path = os.path.join(test_emotion_path, filename)
            img = imread(img_path)
            test_images.append(img)
            test_labels.append(i)

#convert lists to arrays
train_images = np.array(train_images)
test_images = np.array(test_images)

#Normalize Images
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

#Convert labels to one-hot encoding
num_classes = len(emotion_classes)

#Binary classifiers and corresponding one-hot encoded values
binary_classifiers = []
binary_classifier_labels_train = []
binary_classifier_labels_test = []

test_labels = np.array(test_labels)
print(len(train_labels))
print(len(train_images))
print(len(test_labels))
print(len(test_images))

for i in range(num_classes):
    
    #Create one hot encoded labels for current class (train)
    binary_labels_train = np.array([1 if label == i else 0 for label in train_labels])
    #Convert binary labels to one-hot encoding
    binary_one_hot_train = to_categorical(binary_labels_train, num_classes=2)[:, 0]  # Extract the first column (current class)
    binary_one_hot_train = binary_one_hot_train.reshape((-1, 1))  # Reshape to (num_samples, 1)
    binary_classifier_labels_train.append(binary_one_hot_train)

    #Create one hot encoded labels for current class (test)
    binary_labels_test = np.array([1 if label == i else 0 for label in test_labels])
    #Convert binary labels to one-hot encoding
    binary_one_hot_test = to_categorical(binary_labels_test, num_classes=2)[:, 0]  # Extract the first column (current class)
    binary_one_hot_test = binary_one_hot_test.reshape((-1, 1))  # Reshape to (num_samples, 1)
    binary_classifier_labels_test.append(binary_one_hot_test)


    #Create model - linear stack of layers, add one layer at a time
    model = Sequential()

    #Add 2D convolutional layer - (32 = number of filters), (3, 3 = size of convolutional layer)
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))

    #Reduces spatial dimensions of feature map, retaining features and reducting complexity
    model.add(MaxPooling2D(2, 2))

    #Add 2 more layers
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    #Flatten multi-dimensional output into one-dimensional array
    model.add(Flatten())

    #Fully connected layer is added, captures global patterns
    model.add(Dense(128, activation='relu'))

    #Add output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    #Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #Add binary classifier and one-hot encoded labels to lists
    binary_classifiers.append(model)
    

    #print summary of first binary classifier
    print(f"Binary Classsifier {i + 1} Summary:")
    model.summary()
    print("\n")

#train and evaluate each binary classifier
for i in range(num_classes):
    current_binary_classifier = binary_classifiers[i]
    current_binary_labels_train = binary_classifier_labels_train[i]
    current_binary_labels_test = binary_classifier_labels_test[i]

    # print("Number of test samples:", len(test_images))
    # print("Number of label samples:", len(binary_classifier_labels[i]))


    #train binary classifier
    history = current_binary_classifier.fit(
        train_images, current_binary_labels_train,
        epochs = 20,
        batch_size = 32)

    eval_result = current_binary_classifier.evaluate(test_images, current_binary_labels_test)
    print(f"\nEvaluate results for binary classifier {i+1}:", eval_result)

    print(f"\nTraining history for binary classifier {i+1}:", history.history)
    print('\n')


