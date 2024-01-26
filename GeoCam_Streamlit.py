import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from PIL import Image
import requests
from io import BytesIO



@st.cache_resource
def load_model(url):

    response = requests.get(url)
    response.raise_for_status()
    model_file = BytesIO(response.content)

    # Chargez le modèle à partir du fichier téléchargé
   
    return model_file 

model_url = "https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.1/deepfaune-vit_large_patch14_dinov2.lvd142m.pt"


model_file = load_model(model_url)

sys.path.append("D:\AlchemIA Technology\Projets\FNC\Etudes\DeepFaune") # A remplacer

from predictTools import PredictorImage



LANG = 'en'
maxlag = 20
threshold = 0.5

## RUNNING BATCHES OF PREDICTION
## ONE AT A TIME
## while True:
##     batch, k1, k2, k1seq, k2seq = predictor.nextBatch()
##     if k1 == len(filenames): break
##     print("Traitement du batch d'images "+str(batch)+"\n")
## OR ALL TOGETHER
# predictor.allBatch()

## GETTING THE RESULTS
## without using the sequences
predictedclass_base, predictedscore_base, best_boxes, count = predictor.getPredictionsBase()
## or using the sequences
predictedclass, predictedscore, best_boxes, count = predictor.getPredictions()

## OUTPUT
dates = predictor.getDates()
seqnum = predictor.getSeqnums()

predictions.to_csv(sys.argv[2], index=False)






# Création de l'interface Streamlit
st.title('TITRE')

# Chargeur d'images
uploaded_file = st.file_uploader("Choisissez une image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption='Image chargée', use_column_width=True)

    # Obtention et affichage des prédictions

    predictor = PredictorImage(filenames, threshold, maxlag, LANG)
    predictor.allBatch()
    predictedclass_base, predictedscore_base, best_boxes, count = predictor.getPredictionsBase()
    predictedclass, predictedscore, best_boxes, count = predictor.getPredictions()

    predictions = pd.DataFrame({'filename':filenames, 'dates':dates, 'seqnum':seqnum, 'predictionbase':predictedclass_base, 
				'scorebase':predictedscore_base, 'prediction':predictedclass, 'score':predictedscore, 'count':count})
    st.write("Prédictions :")
    st.write(predictions)

    # Bouton pour exporter les prédictions
    if st.button('Exporter les Prédictions'):
        # Exportation des prédictions en CSV
        predictions.to_csv('predictions.csv')
        st.success('Les prédictions ont été exportées dans un fichier CSV.')

        # Téléchargement du fichier CSV
        with open('predictions.csv') as f:
            st.download_button('Télécharger le CSV', f, 'predictions.csv')
