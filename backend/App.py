import pandas as pd
import pickle

# Obtener el nombre del café del usuario
cafe_name = input("Introduce el nombre del café: ")

# List of cafes
full_cafes = pd.read_csv("../data/sidewalk-cafe-permits.csv")
name_cafes = pd.read_csv("../data/name_cafes.csv")

# Filtrar los cafés por el nombre ingresado por el usuario
reference_cafes = full_cafes[full_cafes["DOING BUSINESS AS NAME"] == cafe_name]

# Verificar si se encontró el café
if not reference_cafes.empty:
    index_cafe = reference_cafes.index.item()
    name_cafes = name_cafes.iloc[index_cafe]

    # Cargar el modelo
    with open('../model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Obtener recomendaciones
    recommendation = model.kneighbors(name_cafes.values.reshape(1, -1))
    recommended_cafes = full_cafes.iloc[recommendation[1][0][1].item()]["DOING BUSINESS AS NAME"]
    print("The best coffees are: ", recommended_cafes)
else:
    print("El café ingresado no se encontró en la lista.")
