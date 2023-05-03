import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained CNN model
model = load_model('best_model.h5')

# Preprocess input image
def preprocess_image(img_path, input_shape):
    img = image.load_img(img_path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

# Load and preprocess the image
input_image_path = '02282fee-ba3f-460e-91e0-630204ab96c2___RS_LB 5031.jpg'
input_shape = model.input_shape[1:3]
preprocessed_image = preprocess_image(input_image_path, input_shape)

# Classify the image using the model
prediction = model.predict(preprocessed_image)
predicted_class = np.argmax(prediction)

# Print the result
if predicted_class == 0:
    print("The plant is sick.")
elif predicted_class == 1:
    print("The plant is healthy.")
else:
    print("Unknown class.")
