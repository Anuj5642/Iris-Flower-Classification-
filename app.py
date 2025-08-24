import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("🌸 Iris Flower Classification")

iris = load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

# Sidebar inputs
sepal_length = st.sidebar.slider("Sepal length", float(X[:,0].min()), float(X[:,0].max()))
sepal_width  = st.sidebar.slider("Sepal width",  float(X[:,1].min()), float(X[:,1].max()))
petal_length = st.sidebar.slider("Petal length", float(X[:,2].min()), float(X[:,2].max()))
petal_width  = st.sidebar.slider("Petal width",  float(X[:,3].min()), float(X[:,3].max()))

# Prediction
prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
st.write("### Predicted Iris species:", iris.target_names[prediction][0])
