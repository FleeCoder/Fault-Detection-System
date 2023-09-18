#Done By: Mina Nadi Shaker Basily

import urllib
import cv2
import numpy as np
from ultralytics import YOLO

url='http://192.168.1.17:8080/shot.jpg'
model=YOLO('runs/classify/train/weights/best.pt')

while True:
    imgRes=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgRes.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    results=model(img)
    names_dict=results[0].names
    probs=results[0].probs
    
    if probs.top1 == 0:
        print("faulty")
    elif probs.top1 == 1:
        print("good")
    else:
        print("none")



