import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from ultralytics import YOLO
from PIL import Image

from multiprocessing import freeze_support



# Load your custom YOLOv8 model


def preprocess_image(img):
    img = Image.fromarray(img)  # Convert NumPy array to PIL Image
    transform = Compose([
        Resize((640, 640)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input = transform(img).unsqueeze(0)
    return Variable(input, requires_grad=True)

def apply_grad_cam(model, input_image, target_class):
    # Access intermediate feature maps
    feature_maps = None
    gradients = None

    # The index of the layer you want to access, change this to the appropriate index
    target_layer = -1

    def hook_feature(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    def hook_gradient(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # Register forward and backward hooks
    layer_index = 0
    for module in model.model.modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            if layer_index == target_layer:
                module.register_forward_hook(hook_feature)
                module.register_backward_hook(hook_gradient)
                break
            layer_index += 1

    input_var = preprocess_image(input_image)
    output = model.predict(input_image)  # Get detections using the predict method

    if output[0].boxes is not None:
        pred = output[0].boxes.cpu().numpy()  # Get detections
        scores = output[0].probs.max(axis=1)  # Get class scores (maximum class probability)
        target_score = scores.sum()
        model.zero_grad()
        target_score.backward()

        pooled_gradients = F.adaptive_avg_pool2d(gradients, (1, 1))
        weights = pooled_gradients.view(-1)

        combined_map = torch.zeros(feature_maps.shape[2:])
        for i, w in enumerate(weights):
            combined_map += w * feature_maps[0, i, :, :]

        combined_map = F.relu(combined_map).detach().numpy()
        combined_map = cv2.resize(combined_map, (input_image.shape[1], input_image.shape[0]))
        combined_map = np.uint8(255 * combined_map)
        heatmap = cv2.applyColorMap(combined_map, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(input_image, 0.6, heatmap, 0.4, 0)

        return superimposed_img
    else:
        print("No detections found in the image.")
        return input_image



def main():
    # Load the YOLOv8 model
    model = YOLO('best.pt')
    print(model)

    # Load the input image
    input_image = cv2.imread('img.png')

    # Specify the target class (unhealthy plant)
    target_class = 1  # Change this to the appropriate class index for unhealthy plants

    # Apply Grad-CAM
    result = apply_grad_cam(model, input_image, target_class)

    # Save and display the result
    cv2.imwrite('result.jpg', result)
    cv2.imwrite('grad_cam_output.png', result)


if __name__ == '__main__':
    main()
