import streamlit as st
import pandas as pd
import pickle

# Load models
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet',
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM',
    'XGBoost', 'KNN'
]

models = {name: pickle.load(open(f'{name}.pkl', 'rb')) for name in model_names}

results_df = pd.read_csv('model evaluation results.csv')

# Streamlit application
st.title('Model Prediction App')

# Input data
st.header('Input Data')
avg_area_income = st.number_input('Average Area Income')
avg_area_house_age = st.number_input('Average Area House Age')
avg_area_number_of_rooms = st.number_input('Average Area Number of Rooms')
avg_area_number_of_bedrooms = st.number_input('Average Area Number of Bedrooms')
area_population = st.number_input('Area Population')

input_data = [{
    'Avg. Area Income': avg_area_income,
    'Avg. Area House Age': avg_area_house_age,
    'Avg. Area Number of Rooms': avg_area_number_of_rooms,
    'Avg. Area Number of Bedrooms': avg_area_number_of_bedrooms,
    'Area Population': area_population
}]

# Model selection
model_name = st.selectbox('Select a model', model_names)

input_df = pd.DataFrame(input_data)

# Prediction
if st.button('Predict'):
    if model_name in models:
        model = models[model_name]
        prediction = model.predict(input_df)[0]
        st.success(f'The house price predicted is {prediction}')
    else:
        st.error('Model not found')

if st.button('Model Evaluation Results'):
    # Display model evaluation results
    st.header('Model Evaluation Results')
    st.dataframe(results_df)