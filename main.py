
from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np
import cv2


def grad_cam(model, img):
    # Get the last convolutional layer
    last_conv_layer = model.model[-1]

    # Get the class predictions and gradients for the predicted class
    pred_class = np.argmax(model.predict(img)[0])
    gradient_function = K.function([model.input],
                                   [model.output, K.gradients(model.output[:, pred_class], last_conv_layer.output)[0]])
    output, gradients = gradient_function([img])

    # Compute the channel-wise gradient values and average across spatial dimensions
    channel_gradients = np.mean(gradients, axis=(0, 1, 2))

    # Compute the weighted sum of the last convolutional layer using the channel gradients as weights
    activation_map = np.dot(last_conv_layer.output[0], channel_gradients)

    # Apply ReLU activation and normalize the heatmap
    heatmap = np.maximum(activation_map, 0)
    heatmap /= np.max(heatmap)

    return heatmap


# Load the YOLO model from Ultralytics
model = YOLO('best.pt')

# Load an example image
img = cv2.imread('img.jpg')

# Apply Grad-CAM to the image
heatmap = grad_cam(model, img)

# Resize the heatmap to match the size of the input image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# Convert the heatmap to a color map using the Jet color map
heatmap_cm = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

# Overlay the color map onto the input image using alpha blending
output = cv2.addWeighted(img, 0.5, heatmap_cm, 0.5, 0)

# Display the input image, heatmap, and output image
cv2.imshow('Input', img)
cv2.imshow('Heatmap', heatmap_cm)
cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
