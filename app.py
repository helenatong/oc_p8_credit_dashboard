############################## Importation de librairies ####################################
import streamlit as st
import requests
import numpy as np
import pandas as pd 
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



##############################################################################################
##############################################################################################
############################## Configurations de l'application ###############################
st.set_page_config(layout="wide")



##############################################################################################
##############################################################################################
############################## Variables globales ############################################
LIST_REFERENCE_GROUP_BASE = ['AGE_GROUP', 'YEARS_EMPLOYED_GROUP', 'GENDER',
                                'NAME_EDUCATION_TYPE_Higher education'] 
LIST_FEATURES_SELECTION_BASE = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1',
                                'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN', 
                                'CC_CNT_DRAWINGS_CURRENT_MAX',
                                'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
                                'YEARS_EMPLOYED' ]
                                # 'BURO_CREDIT_ACTIVE_Closed_MEAN', 
                                # 'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN']
MAX_LABEL = [i for i in range (0, 16)]
API_URL = "https://apigrantcredit2-ddccfad9cpc3akcx.westeurope-01.azurewebsites.net/predict"
DEBUG_MODE = False



##############################################################################################
##############################################################################################
############################## PARTIE FONCTIONS ##############################################

##############################################################################################
############################## FONCTIONS GRAPHIQUES ##########################################
@st.cache_data
def gauge_chart(value):
    """
    Affiche la valeur value sous forme graphique avec repères
    """
    # Définition de la couleur selon la valeur
    if value <= 0.35:
        color = "#FF0000"  # Rouge
    elif value < 0.53:
        color = "#FFA500"  # Orange
    else:
        color = "#00FF00"  # Vert
    
    # CSS pour la barre et les repères
    st.markdown(f"""
    <style>
        .stProgress > div > div > div > div {{
            background-color: {color};
        }}
        .gauge-container {{
            position: relative;
            width: 100%;
            height: 30px;
        }}
        .gauge-marker {{
            position: absolute;
            height: 30px;
            top: 25px;
            border-left: 2px solid;
        }}
        .gauge-label {{
            position: absolute;
        }}
    </style>
    <div class="gauge-container">
        <div class="gauge-label" style="left: 0;">Non accordé: 0</div>
        <div class="gauge-label" style="right: 0;">Accordé: 1</div>
        <div class="gauge-marker" style="left: 53%; border-color: red;"></div>
        <div class="gauge-label" style="left: 53%; transform: translateX(-50%);">Seuil: 0.53</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(value)

@st.cache_data
def generate_shap_plot(_shap_values, feature_names_shap, \
                        max_display_global, max_display_local, id):
    """
    Retourne les figures SHAP (Global et Local)
    """
    # Global SHAP
    global_fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(_shap_values,
                        max_display=max_display_global, 
                        feature_names=feature_names_shap, 
                        show=False)
    plt.close()

    # Local SHAP
    local_fig = plt.figure(figsize=(10, 6))
    shap.plots.bar(_shap_values[id],
                    max_display=max_display_local)
    plt.close()
    return(global_fig, local_fig)

@st.cache_data
def generate_box_plot(data, x=None, y=None, hue=None):
    """
    Retourne la figure box plot
    """
    fig = plt.figure(figsize=(7, 4))
    sns.boxplot(data=data, 
                x=x, 
                y=y, 
                hue=hue)
    plt.close()
    return(fig)

@st.cache_data
def generate_scatter_plot(data, x=None, y=None, hue=None, highlight_point=None, highlight_color='Black'):
    """
    Retourne la figure scatter plot
    """
    fig = plt.figure(figsize=(7, 4))
    sns.scatterplot(data=data, 
                    x=x, 
                    y=y, 
                    hue=hue)
    if highlight_point is not None:
        plt.scatter(highlight_point[0], highlight_point[1], 
                    color=highlight_color, 
                    marker='x',
                    s=300,  # Taille de la croix
                    label='Client',
                    zorder=5) # Au dessus de tous les points
    plt.close()
    return(fig)

##############################################################################################
############################## FONCTIONS LIEES A LA DONNEE ###################################
@st.cache_data
def load_df():
    """
        Charge le df test avec les valeurs brutes et les prédictions TARGET
    """
    df_original = pd.read_parquet("data/df_test_processed.pq")
    return(df_original)

@st.cache_data
def get_ids(df):
    """
        Retourne la liste de tous les id disponible dans la base de données
    """
    return(df['SK_ID_CURR'].values.tolist())

@st.cache_data
def get_features(df):
    """
        Retourne la liste des features explicatives
    """
    all_variables = df.columns.to_list()
    return([feature for feature in all_variables if feature not in ['SK_ID_CURR', 'TARGET']])

@st.cache_data
def create_new_columns(df):
    """
        Ajouter des colonnes
    """
    # Ajout d'une classe âge
    df['AGE'] = df['DAYS_BIRTH'].apply(lambda x: int(-x/365))
    cut_off = [i for i in range(20, 71, 5)]
    df['AGE_GROUP'] = pd.cut(x=df['AGE'], bins=cut_off)

    # Ajout d'une classe nombre d'années travaillées
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: x if np.isnan(x) else round((-x/365), 1))
    cut_off = [0, 1, 2, 3, 5, 8, 15, 20, 30, 50]
    df['YEARS_EMPLOYED_GROUP'] = pd.cut(x=df['YEARS_EMPLOYED'], bins=cut_off)

    # Renommer
    df['GENDER'] = df['CODE_GENDER'].apply(lambda x: 'Male' if x==1 else 'Female')
    df['TARGET'] = df['TARGET'].apply(lambda x: 'Grant' if x==1 else 'Refuse')
    return (df)

@st.cache_resource # Cache pour les modèles
def load_shap_model(feature_names_shap):
    """
        Chargement du modèle Explainer SHAP
    """
    shap_values = joblib.load("model/shap_values.joblib")
    shap_values.feature_names = feature_names_shap
    return(shap_values)

@st.cache_data
def get_api_response(id, API_URL=API_URL):
    response = requests.post(API_URL, params={'id': id})
    result = response.json()
    return(response, result)

@st.cache_data
def load_df_id(df, id):
    """
        Charge le df test avec les valeurs brutes et les prédictions TARGET du client
    """
    return(df.loc[df['SK_ID_CURR'] == id])

##############################################################################################
##############################################################################################
############################## CHARGEMENT DES DONNEES ########################################
df_original = load_df()
feature_names_shap = get_features(df_original) 
shap_values = load_shap_model(feature_names_shap)
df_original = create_new_columns(df_original)
LIST_ID = get_ids(df_original)



##############################################################################################
##############################################################################################
############################## CODE ##########################################################

##############################################################################################
############################## SIDE BAR ######################################################
with st.sidebar:
    st.header('Bienvenue 👋')

    st.subheader("Id du client:")
    chosen_id = st.selectbox(label="Entrez l'id du client",
                             options=LIST_ID,
                            label_visibility="collapsed")
    if chosen_id:
        if chosen_id not in LIST_ID:
            st.warning("L'ID choisi n'existe pas.")
            st.stop()
        shap_index = df_original[df_original['SK_ID_CURR']==chosen_id].index[0]

    data_display = st.checkbox(label="Données brutes du client")
    
    st.subheader("SHAP - Nombre de features à afficher:")
    nb_label_local = st.select_slider(label="SHAP Local",
                                        value=10,
                                        options=MAX_LABEL)
    nb_label_global = st.select_slider(label="SHAP Global",
                                        value=6,
                                        options=MAX_LABEL)

    st.subheader("Significations de variables clées")
    st.write("EXT_SOURCE_i : Score calculé par d'autres structures, détails indisponibles"

##############################################################################################
############################## HEADER - PREDICTION CLIENT ####################################
col1_Lspace, col1, col1_Rspace = st.columns([0.1, 0.5, 0.1]) #pour centrer
with col1:
    st.title("Dashboard de Prêt Bancaire")
    if not chosen_id:
        st.subheader("👈 :red[Choisissez un id]")

    # Récupérer la prédiction du client
    col1a, col1b, col1c = st.columns([0.2, 0.4, 0.4])
    if chosen_id:
        response, result = get_api_response(chosen_id, API_URL)
        if response.status_code == 404:
            st.write(result['detail'])
        else:
            if DEBUG_MODE: st.write(result)
            proba = result['probabilité_de_remboursement']
            with col1a:
                st.subheader(f'**Résultat:**')
            with col1b:
                if proba <= 0.5: 
                    st.write('')
                    st.write(proba, f":red[{result['prediction']}]")
                else: st.write(proba, f":green[{result['prediction']}]")
            with col1c:
                gauge_chart(proba)
    
    df_id_raw = load_df_id(df_original, chosen_id)
    if data_display:
        st.subheader('Données brutes client: ')
        st.dataframe(data=df_id_raw, 
                        hide_index=True)
        

##############################################################################################
############################## SHAP CHARTS - GLOBAL & LOCAL ##################################
col21, col22 = st.columns(2)
if chosen_id:
    global_fig, local_fig = generate_shap_plot(
                                _shap_values=shap_values, 
                                feature_names_shap=feature_names_shap, 
                                max_display_global=nb_label_global, 
                                max_display_local=nb_label_local, 
                                id=shap_index)
    with col21:
        st.subheader("Shap Global")
        st.pyplot(global_fig)

    with col22:
        st.subheader("Shap Local")
        st.pyplot(local_fig)


##############################################################################################
############################## GRAPHIQUE - ANALYSE UNIVARIEE BOX PLOT ########################
if chosen_id:
    st.subheader("**Distribution des variables quantitatives**")
    c1b, c2b, c3b = st.columns([0.3, 0.5, 0.2])
    with c1b:
        st.write('Choisissez les features à afficher:')
        LIST_FEATURES_SELECTION = [col for col in LIST_FEATURES_SELECTION_BASE \
                                        if not np.isnan(df_id_raw.loc[:, col].values[0])]
    with c2b:
        selected_features = st.multiselect(label='Features', 
                                            options=LIST_FEATURES_SELECTION, 
                                            label_visibility="collapsed")
    if selected_features:
        c1i, c2i, c3i = st.columns([0.3, 0.5, 0.2])
        with c1i:
            st.write('Valeurs du client:')
        with c2i:
            st.dataframe(data=df_original.loc[df_original['SK_ID_CURR']==chosen_id, \
                                                                        selected_features], 
                        hide_index=True)
        col4a_Lspace, col4a, col4a_Rspace = st.columns([0.1, 0.8, 0.1])
        with col4a:
            box_fig1 = generate_box_plot(data=df_original[selected_features])
            st.pyplot(box_fig1)


##############################################################################################
############################## GRAPHIQUE - ANALYSE BIVARIEE BOX PLOT #########################
if chosen_id:
    st.subheader("**Analyse bivariée - Boxplot**")
    c1a, c2a, c3a = st.columns([0.3, 0.5, 0.2])
    with c1a:
        st.write('Choisissez une feature quantitative:')
    with c2a:
        selected_featureB1 = st.selectbox(label='FeatureB1', 
                                        options=LIST_FEATURES_SELECTION, 
                                        label_visibility="collapsed",
                                        index=None)
    
    c1c, c2c, c3c = st.columns([0.3, 0.5, 0.2])
    with c1c:
        st.write('Choisissez le groupe de référence:')
        LIST_REFERENCE_GROUP_SELECTION = [col for col in LIST_REFERENCE_GROUP_BASE \
                                        if not pd.isna(df_id_raw.loc[:, col].values[0])]
    with c2c:
        reference_group = st.selectbox(label='Ref Group', 
                                        options=LIST_REFERENCE_GROUP_SELECTION, 
                                        label_visibility="collapsed",
                                        index=None)

    if selected_featureB1 and reference_group:
        c1h, c2h, c3h = st.columns([0.3, 0.5, 0.2])
        with c1h:
            st.write('Valeurs du client:')
        with c2h:
            client_data = df_original.loc[df_original['SK_ID_CURR'] == chosen_id, \
                                                    [selected_featureB1,reference_group]]
            st.dataframe(data=client_data, 
                        hide_index=True)
        col3a_Lspace, col3a, col3a_Rspace = st.columns([0.1, 0.6, 0.1])
        with col3a:
            box_fig2 = generate_box_plot(data=df_original, 
                                        x=reference_group, 
                                        y=selected_featureB1, 
                                        hue='TARGET')
            st.pyplot(box_fig2)


##############################################################################################
############################## GRAPHIQUE - ANALYSE BIVARIEE SCATTER PLOT #####################
if chosen_id:
    st.subheader("**Analyse bivariée - Scatter plot**")
    c1d, c2d, c3d = st.columns([0.3, 0.5, 0.2])
    with c1d:
        st.write('Choisissez deux features quantitatives:')
    with c2d:
        select_feature_SC = st.multiselect(label='select_feature_SC', 
                                    options=LIST_FEATURES_SELECTION, 
                                    label_visibility="collapsed", 
                                    max_selections=2)

    c1e, c2e, c3e = st.columns([0.3, 0.5, 0.2])
    with c1e:
        st.write('Choisissez la classe:')
    with c2e:
        LIST_CLASSES = LIST_REFERENCE_GROUP_BASE
        LIST_CLASSES.append('TARGET')
        chosen_class = st.selectbox(label='Ref Group', 
                                    options=LIST_CLASSES, 
                                    label_visibility="collapsed",
                                    index=None)

    c1f, c2f, c3f = st.columns([0.3, 0.5, 0.2])  
    with c1f:
        st.write('Choisissez le nombre de points:')
    with c2f:
        nb_sample = st.number_input(label='nb individu', 
                                    label_visibility='collapsed', 
                                    min_value=1000, 
                                    max_value=df_original.shape[0], 
                                    value="min")

    if chosen_class and len(select_feature_SC) == 2 and nb_sample:
        c1g, c2g, c3g = st.columns([0.3, 0.5, 0.2])  
        with c1g:
            st.write('Valeurs du client:')
        with c2g:
            df_values = df_original.loc[df_original['SK_ID_CURR'] == chosen_id, \
                                            [select_feature_SC[0], select_feature_SC[1], chosen_class]]
            st.dataframe(data=df_values, hide_index=True)
        col3a_Lspace, col3a, col3a_Rspace = st.columns([0.1, 0.6, 0.1])
        with col3a:
            scatter_fig = generate_scatter_plot(data=df_original.sample(nb_sample), 
                                                x=select_feature_SC[0], 
                                                y=select_feature_SC[1], 
                                                hue=chosen_class,
                                                highlight_point=df_original.loc[df_original['SK_ID_CURR'] == chosen_id].iloc[0],
                                                highlight_color='Black')
            st.pyplot(scatter_fig)
