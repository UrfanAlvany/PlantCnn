import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tf_explain.core.grad_cam import GradCAM
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2


def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def main():
    model_path = 'best_model.h5'
    img_path = '0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.jpg'
    target_size = (224, 224)

    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the input image
    img_array = load_and_preprocess_image(img_path, target_size)

    # Make a prediction with the model
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])

    # Apply Grad-CAM
    explainer = GradCAM()

    grid_cam_result = explainer.explain((img_array, None), model, class_idx, 'conv2d_2')
    # Replace 'conv2d_N' with the name of the last convolutional layer in your model

    # Create a custom colormap with blue and red
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'red'], N=256)

    # Normalize the Grad-CAM heatmap
    heatmap = (grid_cam_result - grid_cam_result.min()) / (grid_cam_result.max() - grid_cam_result.min())

    # Apply the custom colormap
    heatmap_colored = cmap(heatmap)

    # Convert the colored heatmap to BGR format
    heatmap_bgr = cv2.cvtColor((heatmap_colored[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Load the original image
    original_image = cv2.imread(img_path)
    original_image = cv2.resize(original_image, target_size)

    # Overlay the heatmap on the original image
    overlay = cv2.addWeighted(heatmap_bgr, 0.6, original_image, 0.4, 0)

    # Save the Grad-CAM result
    cv2.imwrite('grad_cam_result.jpg', overlay)


if __name__ == '__main__':
    main()
