import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App Title
st.title(" Iris Flower Prediction App ")
st.write("Adjust the sliders to predict the Iris species.")

# Sidebar for User Inputs
st.sidebar.header("Input Features")


def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)

    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# Display User Input
st.subheader("User Input parameters")
st.write(df)

# Prediction Logic
if st.button("Predict"):
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    species = ['Setosa', 'Versicolor', 'Virginica']

    st.subheader('Prediction')
    st.success(f"The predicted species is: **{species[prediction[0]]}**")

    st.subheader('Prediction Probability')
    st.write(pd.DataFrame(prediction_proba, columns=species))