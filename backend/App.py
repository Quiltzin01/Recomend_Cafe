import pandas as pd
import pickle

# cafe of the user
cafe_name = "SMALL CHEVAL"

# list of cafes
full_cafes = pd.read_csv("../data/sidewalk-cafe-permits.csv")
name_cafes = pd.read_csv("../data/name_cafes.csv")

reference_cafes = full_cafes[full_cafes["DOING BUSINESS AS NAME	"] == full_cafes]
index_cafe = reference_cafes.index.item()
name_cafes = name_cafes.iloc[index_cafe]

# load model
with open('../', 'rb') as f:
    model = pickle.load(f)

recommendation = model.kneighbors(name_cafes.values.reshape(1, -1))
recommended_cafes = full_cafes.iloc[recommendation[1][0][1].item()]["DOING BUSINESS AS NAME	"]
print("The best coffees are: ", recommended_cafes)