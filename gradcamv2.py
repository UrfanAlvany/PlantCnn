import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tf_explain.core.grad_cam import GradCAM
import cv2


def overlay_heatmap_on_image(heatmap, original_image, alpha=0.5):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype("uint8"), cv2.COLORMAP_JET)
    original_image_uint8 = original_image.astype("uint8")
    overlay_result = cv2.addWeighted(heatmap_color, alpha, original_image_uint8, 1 - alpha, 0)
    return overlay_result


def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def main():
    model_path = 'sick_vs_healthy_model.h5'
    img_path = '00a7c269-3476-4d25-b744-44d6353cd921___GCREC_Bact.Sp 5807.jpg'
    target_size = (150, 150)

    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the input image
    img_array = load_and_preprocess_image(img_path, target_size)

    # Make a prediction with the model
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])

    # Apply Grad-CAM
    explainer = GradCAM()
    grid_cam_result = explainer.explain((img_array, None), model, class_idx, 'conv2d_3')

    # Resize the heatmap
    grid_cam_result = cv2.resize(grid_cam_result, (224, 224))

    # Save the Grad-CAM result
    image.save_img('grad_cam_result.jpg', grid_cam_result)

    # Load original image
    original_image = image.load_img(img_path)
    original_image = image.img_to_array(original_image)

    # Overlay the heatmap on the original image
    overlay_result = overlay_heatmap_on_image(grid_cam_result, original_image, alpha=0.5)

    # Save the overlay result
    image.save_img('overlay_result.jpg', overlay_result)


if __name__ == '__main__':
    main()
