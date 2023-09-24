# Importation des bibliothèques nécessaires
import numpy as np
import cv2
import streamlit as st
from deepface import DeepFace
import gc
import av
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer

# Décorateur pour l'analyse d'images
def analyzer(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper

# Fonction pour redimensionner une image
def resizing_images(image):
    width = 480
    heigh = int((width * image.shape[0])/ image.shape[1])
    dimension = (width, heigh)
    return cv2.resize(image, dimension,interpolation = cv2.INTER_AREA)

# Fonction pour détecter les visages dans une image
@analyzer
@st.cache_resource
def detecting_faces(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_frontalface_default.xml')
    return detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                      minSize=(150,150))

# Fonction pour extraire l'émotion d'une image
@analyzer
@st.cache_resource
def Extract_emotion(image):
    result = DeepFace.analyze(image,['emotion'],
                             detector_backend="ssd", enforce_detection=False)
    return str(result[0]["dominant_emotion"])

# Fonction pour la reconnaissance faciale
@analyzer
@st.cache_resource
def face_Recognition(image, Path):
    result = DeepFace.verify(image, Path, distance_metric="euclidean_l2",
                           detector_backend="ssd",enforce_detection=False)
    return bool(result["verified"])

# Fonction pour encadrer en vert
def green_Rectangle(image, x, y, w, h, emotion):
    cv2.rectangle(image, (x, y), (x + w, y + h), (10, 200, 10),
                  thickness=int( image.shape[0] * 0.005), lineType=cv2.LINE_AA)
    cv2.rectangle(image, (x, y), (x + w, y - int(image.shape[1] / 40)),
                      (200, 200, 200), cv2.FILLED)
    cv2.putText(image, emotion , (x , y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (40, 40, 40))

# Fonction pour encadrer en rouge
def red_Rectangle(image, x, y, w, h, emotion):
    cv2.rectangle(image, (x, y), (x + w, y + h), (10, 20, 210),
                  thickness=int( image.shape[0] * 0.005), lineType=cv2.LINE_AA)
    cv2.rectangle(image, (x, y), (x + w, y - int(image.shape[1] / 40)),
                      (200, 200, 200), cv2.FILLED)
    cv2.putText(image, emotion , (x , y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (40, 40, 40))

# Fonction pour détecter l'émotion dans une image
def image_emotion(image):
    image = resizing_images(image)
    faces = detecting_faces(image)
    emotion = Extract_emotion(image)
    for x, y, w, h in faces:
        green_Rectangle(image, x, y, w, h, emotion)
    return image, emotion

# Fonction pour détecter l'émotion en temps réel
def stream_emotion(frame):
    frame = frame.to_ndarray(format="bgr24")
    frame = resizing_images(frame)
    frame = cv2.flip(frame, 1)
    emotion = Extract_emotion(frame)
    faces = detecting_faces(frame)
    find = face_Recognition(frame, 'user.jpg')
    for x, y, w, h in faces:
        if find:
            green_Rectangle(frame, x, y, w, h, f'l\'utilisateur est {emotion}')
        else:
            red_Rectangle(frame, x, y, w, h, f'une personne est {emotion}')
        gc.collect()
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

# Fonction pour le traitement d'image statique
def image():
    image = st.file_uploader("Choisir une image 📷...", type=['jpg', 'png', 'jpeg'])
    if image is not None:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
        image, emotion = image_emotion(image)
        st.image(image=image, caption=f'L\'émotion détectée est {emotion}', channels='BGR')

# Fonction pour définir l'utilisateur
def user():
    st.header("Définir l'utilisateur")
    picture = st.camera_input("Prendre une photo de l'utilisateur")
    if picture:
        pil_image = Image.open(picture).convert('RGB')
        numpy_image = np.array(pil_image)
        final_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        folder_path = "C:/Users/DALL/Project Emotion detection"  # Chemin du dossier où enregistrer l'image
        file_name = "user.jpg"  # Nom du fichier image
        cv2.imwrite(os.path.join(folder_path, file_name), final_image)

# Fonction principale
def main():
    st.image('images ressource/norsys logo.png', caption="Norsys Afrique")
    st.title('Détection des émotions 💻')
    st.sidebar.title('Navigation')
    select = st.sidebar.radio("Choisir un mode", ["À propos de l'application", "Détection en temps réel", "Détection en image", "Identification de l'utilisateur"])

    if select == "Détection en temps réel":
        Live()
    elif select == "Détection en image":
        image()
    elif select == "À propos de l'application":
        about()
    else:
        user()

if __name__=="__main__":
    main()
