# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import time
import cv2
import os
from smbus2 import SMBus
# from mlx90614 import MLX90614
import screeninfo


def detect(imageList, locsList, predsList, predsList2, predsTfList, tempList, predsImageList):
    # bus = SMBus(1)
    # sensor = MLX90614(bus, address=0x5A)
    maskNet = load_model("mask_detector.model")
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

    def detect_face(frame):
        (h, w) = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(35, 35)
        )
        faces = []
        faces_tf = []
        locs = []
        preds = [[0, 0]]
        for (x, y, w, h) in det_faces:

            startX = x
            startY = y
            endX = w
            endY = h
            locs.append((startX, startY, endX, endY))
            face = frame[startY:startY + endY, startX:startX+endX]
            faces.append(face)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces_tf.append(face)

        return locs, faces, faces_tf

    def detect_mask(faces, maskNet):
        preds = None
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=1)[0]
        return preds

    count = 0
    label = None
    color = (0, 0, 255)
    img = cv2.imread("man.jpg")
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25,
                             interpolation=cv2.INTER_AREA)
    locs, faces, faces_tf = detect_face(small_frame)
    preds_tf = detect_mask(faces_tf, maskNet)
    time.sleep(3)
    count = 0
    (targetStartX, targetEndX, targetStartY, targetEndY) = (150, 340, 140, 350)
    threshold = 2000

    while True:
        if not imageList.empty():
            count += 1
            image = imageList.get()
            small_frame = cv2.resize(image, (0, 0), fx=0.25,
                                     fy=0.25, interpolation=cv2.INTER_AREA)
            locs, faces, faces_tf = detect_face(small_frame)
            locsList.put(locs)
            for box in locs:
                (startX, startY, endX, endY) = box
                upperCorner = startX*4, startY*4
                bottomCorner = (startX + endX)*4, (startY + endY)*4
                mse = (startX*4 - targetStartX)**2 + ((startX + endX)*4 - targetEndX)**2 + \
                    (startY*4 - targetStartY)**2 + \
                    ((startY + endY)*4 - targetEndY)**2
                if (mse < threshold) and count % 30 == 0:
                    preds_tf = detect_mask(faces_tf, maskNet)
                    predsTfList.put(preds_tf)
                    # sensor.get_obj_temp(),1))
                    tempList.append(round(np.randint(30, 40)))
                if len(tempList) > 5:
                    tempList.pop(0)
