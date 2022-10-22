import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import csv

st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(page_title="SeAI", page_icon="./logo.png")

file = './data_map.csv'


@st.cache()
def prediction(image, model):
    [shape] = model.get_layer(index=0).input_shape
    size = shape[1:-1]
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = tf.keras.preprocessing.image.smart_resize(input_arr, size)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    [predictions] = tf.keras.applications.imagenet_utils.decode_predictions(predictions, 2)
    return predictions

#Fonction pour formater l'image avant de prédire

@st.cache()
def load_model():
    return ResNet50(weights='imagenet')
model = load_model()
#Chargement du modèle
#Le modèle => ResNet50

a = st.sidebar.selectbox('Navigation:', ["Upload de l'image", "Carte", "Statistiques"])

#-------------------------------------------------------------------------------------------------------

if a == "Upload de l'image":
    st.title("Uploadez une image :")
    uploaded_file = st.file_uploader("Choisissez une image", type=['png', 'jpg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)
        predictions = prediction(image, model)
        radio_pred = []
        for x in predictions:
            [name, description, score] = x
            radio_pred.append(description)
        radio_pred.append("autre")
        pred = st.radio('Select', radio_pred)

        if pred == "autre":
            final_pred = st.text_input('Entrez la bonne catégorie')
        else:
            final_pred = pred

        "votre choix est : ", final_pred

        loc_button = Button(label="Valider")
        loc_button.js_on_event("button_click", CustomJS(code="""
            navigator.geolocation.getCurrentPosition(
                (loc) => {
                    document.dispatchEvent(new CustomEvent("GET_LOCATION", {detail: {lat: loc.coords.latitude, long: loc.coords.longitude}}))
                }
            )
            """))

        result = streamlit_bokeh_events(
            loc_button,
            events="GET_LOCATION",
            key="get_location",
            refresh_on_update=False,
            override_height=75,
            debounce_time=0)

        if result:
            st.dataframe(result)
            new_data = pd.DataFrame({
                'category': [final_pred],
                'lat': [result['GET_LOCATION']['lat']],
                'long': [result['GET_LOCATION']['long']]
            })
            
            data_map = pd.read_csv(file, index_col=0)
            data_map = data_map.append(new_data, ignore_index=True)
            
            data_map.to_csv(file)

#Lorsqu'on clique sur le bouton "Upload de l'image"

if a == "Carte":
    st.title("Carte")
    data_map = pd.read_csv(file, index_col=0)
    data_map

    m = folium.Map(location=[46.232192999999995, 2.209666999999996], zoom_start=6,tiles="Stamen Terrain")
    data_map = pd.read_csv(file, delimiter=",")
    marker_cluster = MarkerCluster().add_to(m)
    for row in data_map.iterrows():
        folium.Marker(
            location=[row[1][2], row[1][3]],
            popup=str(row[1][1]),
            icon=folium.Icon(color="green", icon="trash"),
        ).add_to(marker_cluster)
    folium_static(m)

#Affichage de la carte + du tableau répertoriant tous les déchets

import base64

main_bg = "ocean.jpg"
main_bg_ext = "jpg"

side_bg = "bar.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Style

if a == "Statistiques":
    
    tableau = {}
    f=open('data_map.csv',"r")
    donnee=f.readlines()
    for row in donnee:
        continue
    n=row.split(',')
    max_dechet=n[-4]
    
    st.title('Histogramme des déchets')
    
    
    f.close()
    for i in range(1,len(donnee)):
        if donnee[i].split(',')[1] not in tableau.keys() :
                tableau[donnee[i].split(',')[1]] = 1
        else :
                tableau[donnee[i].split(',')[1]]+=1

    print(tableau)
    fig = plt.figure()

    colonne=tableau.keys()
    valeur=[]
    for elem in colonne :
        valeur.append(tableau[elem])
    largeur = 0.9
    
    plt.bar(colonne,valeur,largeur,color='#1E90FF')
    plt.title('Quantité de déchets par catégorie')
    plt.xlabel('Catégories')
    plt.ylabel('Quantité')
    plt.xticks(rotation=90)
    fig.suptitle('Nombre total de déchets traités :'+ ' ' +max_dechet, fontsize=22)
    st.pyplot(fig)

    print(tableau)

#Affichage du diagramme avec les statistiques