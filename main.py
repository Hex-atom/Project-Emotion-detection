import numpy as np
import cv2
import streamlit as st
from deepface import DeepFace
import gc
import av
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# Decorator for analysis functions
def analyzer(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper


# Function to resize images
def resizing_images(image):
    width = 480
    heigh = int((width * image.shape[0])/ image.shape[1])
    dimension = (width, heigh)
    return cv2.resize(image, dimension,interpolation = cv2.INTER_AREA)


# Function to detect faces
@analyzer
@st.cache_resource
def detecting_faces(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_frontalface_default.xml')
    return detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                      minSize=(150,150))


# Function to extract emotion
@analyzer
@st.cache_resource
def Extract_emotion(image):
    result = DeepFace.analyze(image,['emotion'],
                             detector_backend="ssd", enforce_detection=False)
    return str(result[0]["dominant_emotion"])
    

# Function for face recognition
@analyzer
@st.cache_resource
def face_Recognition(image, Path):
    result = DeepFace.verify(image, Path, distance_metric="euclidean_l2",
                           detector_backend="ssd",enforce_detection=False)
    
    return bool(result["verified"])


# Function to draw green rectangle
def green_Rectangle(image, x, y, w, h, emotion):
    cv2.rectangle(image, (x, y), (x + w, y + h), (10, 200, 10),
                  thickness=int( image.shape[0] * 0.005), lineType=cv2.LINE_AA)
    cv2.rectangle(image, (x, y), (x + w, y - int(image.shape[1] / 40)),
                      (200, 200, 200), cv2.FILLED)
    cv2.putText(image, emotion , (x , y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (40, 40, 40))


# Function to draw red rectangle
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

def image():
    image = st.file_uploader(" Choisir une image 📷...",type=['jpg','png','jpeg'])
    if image is not None:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
        image,emotion = image_emotion(image)
        st.image(image= image, caption= f' l\'emotion detecter est {emotion}', channels = 'BGR',)

def user():
    
    st.header('difinir l\'utilisateur')

    picture = st.camera_input("Prendre une photo de l\'utilisateur ")
    if picture:
        pil_image = Image.open(picture).convert('RGB')
        
        numpy_image = np.array(pil_image)
        final_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        folder_path = "C:/Users/DALL/Project Emotion detection"
        file_name = "user.jpg"
        cv2.imwrite(os.path.join(folder_path, file_name), final_image)


def Live():
    st.header("La détection d'émotions en temps réel. 📽 ")

    webrtc_streamer(key="Live stream ",  mode=WebRtcMode.SENDRECV,
                                 video_frame_callback=stream_emotion,
                                   media_stream_constraints={"video": True, "audio": False},
                                        async_processing=True,)

def about():
    st.header("À propos de l'application 🕵🏼‍♂️")
    st.write("L'application de détection d'émotions, alimentée par Streamlit et DeepFace, offre"
              + " une plateforme intuitive aux utilisateurs pour détecter facilement les émotions sur les visages.\n ")
    st.write("Que ce soit en téléchargeant des images ou en utilisant le flux de la webcam, l'application utilise "
              + " des techniques de pointe en apprentissage automatique pour analyser les expressions faciales,"
              + " révélant une gamme d'émotions telles que la joie, la tristesse, la colère, et bien plus encore. \n")
    st.write("Avec une interface simple, les utilisateurs peuvent passer facilement entre les modes image et temps réel,"
              + " assistant à l'analyse des émotions en direct. De plus,"
              + " l'application permet la vérification des visages par rapport à des images de référence, offrant des informations précieuses \n ")
    st.write("Cette application web interactive illustre la synergie entre l'IA et l'engagement des utilisateurs"
              + " en en faisant le choix parfait pour ceux qui sont intrigués par l'apprentissage automatique et les expériences web dynamiques")



def main():
    st.image('images ressource/norsys logo.png',caption="Norsys Afrique",)
    st.title('detection des emotion 💻')
    st.sidebar.title('Navigation')
    select = st.sidebar.radio("Choisir Un mode",["à propos de l'application","détection en temps réel", "détection en image","identification de l'utilisateur"])
    
    if select == "détection en temps réel":
        Live()
    elif select == "détection en image":
        image()
    elif select == "à propos de l'application":
        about()
    else:
        user()

if __name__=="__main__":
    main()