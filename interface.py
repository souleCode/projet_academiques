import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


# Charger le modèle
model = tf.keras.models.load_model('models/model2.h5')

# Correspondance classe-numéro dans vos modèles
class_indices = {
    0: 'daisy',
    1: 'dandelion',
    2: 'roses',
    3: 'sunflowers',
    4: 'tulips'
}

# Fonction pour prétraiter l'image


def preprocess_image(image, target_size):
    # Redimensionner l'image
    img = image.resize(target_size)
    img_array = img_to_array(img)
    # Ajouter une dimension supplémentaire
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normaliser l'image
    return img_array

# Fonction pour faire une prédiction avec le premier modèle


def prediction(image, model, class_indices, target_size):
    img_array = preprocess_image(
        image, target_size=target_size)  # Prétraiter l'image
    predictions = model.predict(img_array)  # Faire l'inférence
    confidence = round(max(predictions[0]) * 100, 2)
    # Obtenir l'indice de la classe prédite
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(f"Predicted class: {predictions}")
    # Obtenir le nom de la classe
    predicted_class_name = class_indices[predicted_class]
    return predicted_class_name, confidence


st.title("Classification des fleurs")
st.sidebar.info("Bienvenue sur mon application de classification des fleurs")

option = st.sidebar.selectbox(
    "Qu'est-ce que vous voulez faire?",
    ("Rien à faire",
     "Classification d'images",
     "Video tracking system",
     "Face recognition system"),
)

if option is not None:
    st.sidebar.write("Vous avez sélectionné:", option)

input_img = st.file_uploader("Choisir une image", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Classifier"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.info("Image utilisée...")
            st.image(input_img, use_column_width=True)
        with col2:
            st.info("Résultats...")
            # Ouvrir l'image directement depuis l'uploader
            image_file = Image.open(input_img)
            class_predicted, confidence = prediction(
                image_file, model, class_indices, (150, 150))
            st.write(f"Application a prédit avec {confidence}% que votre fleur est {
                class_predicted}.\n\n Etes vous satisfait?")

        with col3:
            st.info("Merci...")
            st.write(
                "J'espère que cette application vous a été bénéfique... Merci et à la prochaine")
