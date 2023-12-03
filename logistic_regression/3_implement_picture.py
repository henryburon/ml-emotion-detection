
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import time
from skimage.feature import hog
np.set_printoptions(threshold=np.inf)  # Can use this to make it print out entire array
import joblib

##########################################################################################

# The purpose of this file is to actually implement the trained Linear Regression model on images and classify them
# It works by loading in the saved trained linear regression model
# It then loads in the images we want to classify, and converts them to 48x48 grayscale images
# Then, it extracts the features, and with those features, runs the loaded Linear Regression model
# Finally, it writes the classified emotion on the image and saves it into a folder

##########################################################################################

start_time = time.time()

# Currently is loading in a model trained with 7 classes and max 3k training images per class
# loaded_model = joblib.load('logistic_regression/trained_model/2class_3ksamples_77%.pk1')
loaded_model = joblib.load('logistic_regression/trained_model/7class_3ksample_42%.pk1')

emotion_indices = {
    0: 'angry',
    1: 'happy',
    2: 'neutral',
    3: 'sad',
    4: 'disgusted',
    5: 'fearful',
    6: 'surprised'
}

#####################################################################################################
# Load in images, reshape to 48x48 pixels, and make grayscale
#####################################################################################################

folder_path = 'images/unknown/original'  # Replace this with the path to your folder containing images

image_list = []
saved_image_list = []

# Iterate through all images in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        try:
            with Image.open(file_path) as img:
                # Process the image here (e.g., display or perform operations)
                # img.show()  # Opens the image using the default viewer
                img.load()
                saved_image_list.append(img)
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

# Numerical Predictions
predictions = loaded_model.predict(image_features_list)
print(predictions)

# Convert numerical predictions to corresponding emotion
predicted_emotions = [emotion_indices[prediction] for prediction in predictions]
print(predicted_emotions)

#####################################################################################################
# Write emotion on image
#####################################################################################################

def add_text_to_image(image, text_to_add, output_path):
    copied_image = image.copy()

    draw = ImageDraw.Draw(copied_image)

    font = ImageFont.truetype('FreeMono.ttf', 35)

    text_width = len(text_to_add) * 20
    text_height = 20

    image_width, image_height = copied_image.size
    text_position = ((image_width - text_width) // 2, image_height - text_height - 10)  # Offset by 10 pixels from the bottom

    # Define background rectangle coordinates
    background_position = (
        text_position[0] - 5,  # Adjust the x-coordinate to set the background width
        text_position[1] + 5,  # Adjust the y-coordinate to set the background height
        text_position[0] + text_width + 5,  # Adjust the width of the background rectangle
        text_position[1] + text_height + 15  # Adjust the height of the background rectangle
    )

    # Draw a black rectangle as background
    draw.rectangle(background_position, fill="black")

    draw.text(text_position, text_to_add, fill="white", font=font)

    copied_image.save(output_path)

output_folder = "images/unknown/classified"

# Ensure the output folder exists, create it if necessary
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through images in image_list and predicted_emotions
for index, (img, emotion) in enumerate(zip(saved_image_list, predicted_emotions)):
    output_path = f"{output_folder}/image_{index + 1}_emotion_{emotion}.jpg"  # Replace with desired output format

    # Add text (predicted emotion) to the image and save
    add_text_to_image(img, emotion, output_path)






#####################################################################################################
# Time calculation
#####################################################################################################

end_time = time.time()
total_time = end_time - start_time
print(f"Total time {total_time} seconds")
