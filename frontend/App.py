import streamlit as st
import pandas as pd
import pickle

# Cargar los datos y el modelo al iniciar la aplicación
full_cafes = pd.read_csv("../data/sidewalk-cafe-permits.csv")
name_cafes = pd.read_csv("../data/name_cafes.csv")

with open('../model.pkl', 'rb') as f:
    model = pickle.load(f)

# Título de la aplicación
st.title("Recomendador de Cafés")

# Entrada del usuario
cafe_name = st.text_input("Introduce el nombre del café:")

# Botón para obtener recomendaciones
if st.button("Obtener Recomendaciones"):
    if not cafe_name:
        st.error("Por favor, introduce un nombre de café.")
    else:
        # Filtrar los cafés por el nombre ingresado por el usuario
        reference_cafes = full_cafes[full_cafes["DOING BUSINESS AS NAME"] == cafe_name]

        if reference_cafes.empty:
            st.error("El café ingresado no se encontró en la lista.")
        else:
            index_cafe = reference_cafes.index.item()
            name_cafes_row = name_cafes.iloc[index_cafe]

            # Obtener recomendaciones
            recommendation = model.kneighbors(name_cafes_row.values.reshape(1, -1))
            recommended_cafes = full_cafes.iloc[recommendation[1][0][1].item()]["DOING BUSINESS AS NAME"]

            # Mostrar las recomendaciones
            st.success(f"El mejor café recomendado es: **{recommended_cafes}**")
