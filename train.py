from roboflow import Roboflow
rf = Roboflow(api_key="qWDCWeHyfeD6tnCYjpyy")
project = rf.workspace("aast-fzvxy").project("fault-detection-zmmfi")
dataset = project.version(13).download("folder")


import os
from ultralytics import YOLO
model=YOLO("yolov8n-cls.pt")
results=model.train(data=dataset.location, epochs=200)