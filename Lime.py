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

# Create a LIME explainer
explainer = lime_image.LimeImageExplainer()

def lime_heatmap(input_model, image_array, num_samples=1000, hide_color=None):
    def model_predict(images):
        return input_model.predict(images)

    explanation = explainer.explain_instance(image_array[0], model_predict, top_labels=2, hide_color=hide_color, num_samples=num_samples)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    return mask

input_image_path = '02282fee-ba3f-460e-91e0-630204ab96c2___RS_LB 5031.jpg'
original_image = cv2.imread(input_image_path)
input_shape = model.input_shape[1:3]
preprocessed_image = preprocess_image(input_image_path, input_shape)

# Classify the image using the model
prediction = model.predict(preprocessed_image)
label = 'Sick' if np.argmax(prediction) == 0 else 'Healthy'

# Apply LIME to generate the heatmap
heatmap = lime_heatmap(model, preprocessed_image)

# Overlay the heatmap on the original image
original_image_resized = cv2.resize(original_image, input_shape[::-1])  # Resize the image
original_image_rgb = cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
overlayed_image = mark_boundaries(original_image_rgb, heatmap).astype(np.float32)
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
