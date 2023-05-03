import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from ultralytics import YOLO
from multiprocessing import freeze_support



COLORS = np.random.uniform(0, 255, size=(80, 3))
def parse_detections(results):
    print(results[0])
    detections = results.xyxy[0]
    boxes, colors, names = [], [], []
    for i in range(len(detections)):
        confidence = detections[i][4]
        if confidence < 0.2:
            continue
        xmin, ymin, xmax, ymax = map(int, detections[i][:4])
        category = int(detections[i][5])
        color = COLORS[category]
        name = CLASSES[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img


image_path = "DanLeaf2.jpg"
img = cv2.imread(image_path)
img = cv2.resize(img, (640, 640))
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.float32(img) / 255
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)

if __name__ == '__main__':
    model = YOLO('best.pt')
    model.val()
    target_layers = [model.model.model[-2]]
    results = model([rgb_img])
    if results:
        boxes, colors, names = parse_detections(results)
        detections = draw_detections(boxes, colors, names, rgb_img.copy())
        Image.fromarray(detections)
    freeze_support()

