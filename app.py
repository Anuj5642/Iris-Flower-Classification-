import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Streamlit app
st.title("Iris Flower Classification 🌸")
st.write(f"Model Accuracy: {acc:.2f}")

# Input from user
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))

# Prediction
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = clf.predict(features)
predicted_class = target_names[prediction[0]]

st.subheader("Prediction")
st.write(f"The predicted Iris species is **{predicted_class}**.")
