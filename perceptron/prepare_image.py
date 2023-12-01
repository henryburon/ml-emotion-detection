from PIL import Image

# Open the image using Pillow
img = Image.open('/home/henry/Desktop/Classes/EE_475/final_project/images/test/me/henry.jpeg')

# Resize the image to 48x48 pixels
img_resized = img.resize((48, 48))
img_bw = img_resized.convert('L')
# Save the resized image as new.png
img_bw.save('/home/henry/Desktop/Classes/EE_475/final_project/images/test/me/new.jpeg')
