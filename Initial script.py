import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor


# Define the relative path to the data file
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'Data', 'Generated data.csv')

# Load data
data = pd.read_csv(data_path)

# Encode categorical variables
enc_cat = ['Building type', 'Building climate']
enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(data[enc_cat]).toarray())
categories = enc.categories_
new_cols = [f"{col}_{val}" for col, vals in zip(enc_cat, categories) for val in vals]
enc_df.columns = new_cols
data = pd.concat([data, enc_df], axis=1)

# Encode dependent variables
roof_enc = LabelEncoder()
data['Roof Insulation Encoded'] = roof_enc.fit_transform(data['Roof Insulation'])

window_enc = LabelEncoder()
data['Window Glazing Encoded'] = window_enc.fit_transform(data['Window Glazing'])

if 'Wall Insulation' in data.columns:
    wall_enc = LabelEncoder()
    data['Wall Insulation Encoded'] = wall_enc.fit_transform(data['Wall Insulation'])

# Train k-nearest neighbors model
X = data[['Building area'] + list(enc_df.columns)]
y = data[['Wall U value', 'Roof U value', 'Wall Insulation Encoded', 'Roof Insulation Encoded', 'Wall Insulation thickness', 'Roof insulation thickness', 'Window U value', 'Window Glazing Encoded']]
k = 5  # number of neighbors to consider
weights = 'distance'  # use inverse distance weighting for weighted average prediction
model = KNeighborsRegressor(n_neighbors=k, weights=weights).fit(X, y)

# Prompt user to enter independent variable values
building_type = input("Enter the building type (e.g. Single family house): ")
building_climate = input("Enter the building climate (e.g. Mediterranean): ")
building_area = float(input("Enter the building area (in square meters): "))

# Encode user input using the same OneHotEncoder object
enc_input = enc.transform([[building_type, building_climate]]).toarray()
enc_input_df = pd.DataFrame(enc_input, columns=new_cols)

# Predict encoded dependent variables for the user input
user_input = pd.concat([pd.DataFrame({'Building area': [building_area]}), enc_input_df], axis=1)
distances, indices = model.kneighbors(user_input)  # get distances and indices of the k-nearest neighbors
neighbor_y = y.iloc[indices[0]]  # select the dependent variables of the k-nearest neighbors
predicted_enc = neighbor_y.mean()  # use mean to predict encoded dependent variables for the user input

# Decode predicted encoded dependent variables
predicted_wall_insulation = "Not enough data"
predicted_roof_insulation = roof_enc.inverse_transform([int(predicted_enc['Roof Insulation Encoded'])])[0]
predicted_window_glazing = window_enc.inverse_transform([int(predicted_enc['Window Glazing Encoded'])])[0]
predicted_wall_insulation_thickness = predicted_enc['Wall Insulation thickness']
predicted_roof_insulation_thickness = predicted_enc['Roof insulation thickness']

if 'Wall Insulation' in data.columns:
    predicted_wall_insulation_encoded = int(predicted_enc['Wall Insulation Encoded'])
    predicted_wall_insulation = wall_enc.inverse_transform([predicted_wall_insulation_encoded])[0]

# Predict the values of the dependent variables
predicted_enc = model.predict(user_input)

# Decode predicted dependent variables
predicted_wall_u_value = predicted_enc[0][0]
predicted_enc = model.predict(user_input)

# Decode predicted dependent variables
predicted_wall_u_value = predicted_enc[0][0]
predicted_roof_u_value = predicted_enc[0][1]
predicted_wall_insulation_encoded = predicted_enc[0][2]
predicted_roof_insulation_encoded = predicted_enc[0][3]
predicted_wall_insulation_thickness = predicted_enc[0][4]
predicted_roof_insulation_thickness = predicted_enc[0][5]
predicted_window_u_value = predicted_enc[0][6]
predicted_window_glazing_encoded = predicted_enc[0][7]

if 'Wall Insulation' in data.columns:
    predicted_wall_insulation_encoded = int(predicted_enc[0][2])
    predicted_wall_insulation = wall_enc.inverse_transform([predicted_wall_insulation_encoded])[0]
else:
    predicted_wall_insulation = "Not enough data"

predicted_roof_insulation = roof_enc.inverse_transform([int(predicted_roof_insulation_encoded)])[0]
predicted_window_glazing = window_enc.inverse_transform([int(predicted_window_glazing_encoded)])[0]

# Print the predicted values of all dependent variables
print(f"Predicted value of Wall U value: {predicted_wall_u_value}")
print(f"Predicted value of Roof U value: {predicted_roof_u_value}")
print(f"Predicted value of Wall Insulation: {predicted_wall_insulation}")
print(f"Predicted value of Roof Insulation: {predicted_roof_insulation}")
print(f"Predicted value of Wall Insulation thickness: {predicted_wall_insulation_thickness}")
print(f"Predicted value of Roof insulation thickness: {predicted_roof_insulation_thickness}")
print(f"Predicted value of Window U value: {predicted_window_u_value}")
print(f"Predicted value of Window Glazing: {predicted_window_glazing}")



