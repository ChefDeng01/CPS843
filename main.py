
from ultralytics import YOLO
#import cv2
import torch

#setup GPU
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)

# load yolov8 model
model = YOLO('yolov8m.pt')

results = model.train(data="data.yaml", epochs=200)