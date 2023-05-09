import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load your trained CNN model
model = load_model('best_model.h5')

def preprocess_image(img_path, input_shape):
    img = image.load_img(img_path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

def segment_plant(image, lower_hsv, upper_hsv):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask

def apply_mask_to_heatmap(heatmap, mask):
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    masked_heatmap = cv2.bitwise_and(heatmap_uint8, heatmap_uint8, mask=mask)
    return masked_heatmap

# Create a LIME explainer
explainer = lime_image.LimeImageExplainer()

def lime_heatmap(input_model, image_array, num_samples=1000, hide_color=None):
    def model_predict(images):
        return input_model.predict(images)

    explanation = explainer.explain_instance(image_array[0], model_predict, top_labels=2, hide_color=hide_color, num_samples=num_samples)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    return mask

input_image_path = '04877a47-d7e8-461d-b943-00099f3a2a95___Com.G_TgS_FL 0838.jpg'
original_image = cv2.imread(input_image_path)
input_shape = model.input_shape[1:3]
preprocessed_image = preprocess_image(input_image_path, input_shape)

# Segment the plant from the original image
lower_hsv = np.array([25, 50, 50])
upper_hsv = np.array([90, 255, 255])
segmented_image, plant_mask = segment_plant(original_image, lower_hsv, upper_hsv)

# Classify the image using the model
prediction = model.predict(preprocessed_image)
label = 'Sick' if np.argmax(prediction) == 0 else 'Healthy'

# Apply LIME to generate the heatmap
heatmap = lime_heatmap(model, preprocessed_image)

# Resize the plant mask to match the heatmap size
resized_plant_mask = cv2.resize(plant_mask, (heatmap.shape[1], heatmap.shape[0]))

# Apply the resized plant mask to the LIME heatmap
masked_heatmap = apply_mask_to_heatmap(heatmap, resized_plant_mask)

# Overlay the masked heatmap on the original image
original_image_resized = cv2.resize(original_image, input_shape[::-1])  # Resize the image
original_image_rgb = cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
overlayed_image = mark_boundaries(original_image_rgb, masked_heatmap).astype(np.float32)
overlayed_image = (overlayed_image * 255).astype(np.uint8)

# Add text to the image
text_scale = 0.5  # Adjust the text size
text = f"Plant Status: {label}"
cv2.putText(overlayed_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), 2)

# Save the output image
cv2.imwrite('image_with_lime_heatmap.jpg', overlayed_image)

# Display the output image
cv2.imshow('LIME', overlayed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()