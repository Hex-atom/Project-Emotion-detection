import numpy as np
import cv2
import streamlit as st
from deepface import DeepFace
import gc
import av
import os
from PIL import Image

def analyzer(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper


def resizing_images(image):
    width = 480
    heigh = int((width * image.shape[0])/ image.shape[1])
    dimension = (width, heigh)
    return cv2.resize(image, dimension,interpolation = cv2.INTER_AREA)


@analyzer
@st.cache_resource
def detecting_faces(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_frontalface_default.xml')
    return detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                      minSize=(150,150))


@analyzer
@st.cache_resource
def Extract_emotion(image):
    result = DeepFace.analyze(image,['emotion'],
                             detector_backend="ssd", enforce_detection=False)
    return str(result[0]["dominant_emotion"])
    


@analyzer
@st.cache_resource
def face_Recognition(image, Path):
    result = DeepFace.verify(image, Path, distance_metric="euclidean_l2",
                           detector_backend="ssd",enforce_detection=False)
    
    return bool(result["verified"])


def green_Rectangle(image, x, y, w, h, emotion):
    cv2.rectangle(image, (x, y), (x + w, y + h), (10, 200, 10),
                  thickness=int( image.shape[0] * 0.005), lineType=cv2.LINE_AA)
    cv2.rectangle(image, (x, y), (x + w, y - int(image.shape[1] / 40)),
                      (200, 200, 200), cv2.FILLED)
    cv2.putText(image, emotion , (x , y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (40, 40, 40))


def red_Rectangle(image, x, y, w, h, emotion):
    cv2.rectangle(image, (x, y), (x + w, y + h), (10, 20, 210),
                  thickness=int( image.shape[0] * 0.005), lineType=cv2.LINE_AA)
    cv2.rectangle(image, (x, y), (x + w, y - int(image.shape[1] / 40)),
                      (200, 200, 200), cv2.FILLED)
    cv2.putText(image, emotion , (x , y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (40, 40, 40))



def image_emotion(image):
    image = resizing_images(image)
    faces = detecting_faces(image)
    emotion = Extract_emotion(image)
    for x, y, w, h in faces:
        green_Rectangle(image, x, y, w, h, emotion)
    return image,emotion



def stream_emotion(frame):
    frame = frame.to_ndarray(format="bgr24")
    frame = resizing_images(frame)
    frame = cv2.flip(frame, 1)
    emotion = Extract_emotion(frame)
    faces = detecting_faces(frame)
    find = face_Recognition(frame, 'user.jpg')
    for x, y, w, h in faces:
        if find:
            green_Rectangle(frame, x, y, w, h, f'user is {emotion}')
        else:
            red_Rectangle(frame, x, y, w, h, f'someone is {emotion}')
        gc.collect()
    return av.VideoFrame.from_ndarray(frame, format="bgr24")
