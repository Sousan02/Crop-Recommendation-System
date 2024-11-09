import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
DATASET_PATH =  r"D:\Crop_recommendation.csv"

# Load and preview dataset
data = pd.read_csv(DATASET_PATH)

# Streamlit UI
st.title("Crop Recommender App")
st.write("This app helps farmers select the best crop based on soil and environmental conditions.")

# Preview the dataset
st.write("Dataset Preview:")
st.write(data.head())

# Data preprocessing
X = data.drop(columns=['label'])  # Features
y = data['label']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_train_noisy = X_train + np.random.normal(0, 0.1, X_train.shape)
X_test_noisy = X_test + np.random.normal(0, 0.1, X_test.shape)
rf_model = RandomForestClassifier(n_estimators=70, max_depth=4, random_state=42)

rf_model.fit(X_train_noisy, y_train)

# Display accuracy on test data
accuracy = rf_model.score(X_test_noisy, y_test)
st.markdown(f"<h1 style='font-weight:bold; font-size:40px;'>Model Accuracy: {accuracy * 100:.2f}%</h1>", unsafe_allow_html=True)

st.subheader("Distribution of Nitrogen Content")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data['N'], kde=True, color='green', ax=ax)  # Green color for Nitrogen
ax.set_title('Nitrogen Content Distribution')
ax.set_xlabel('Nitrogen Level')
ax.set_ylabel('Frequency')
st.pyplot(fig)

st.subheader("Distribution of Phosphorous Content")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data['P'], kde=True, color='blue', ax=ax)  # Blue color for Phosphorus
ax.set_title('Phosphorous Content Distribution')
ax.set_xlabel('Phosphorous Level')
ax.set_ylabel('Frequency')
st.pyplot(fig)

st.subheader("Distribution of Potassium Content")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data['K'], kde=True, color='orange', ax=ax)  # Orange color for Potassium
ax.set_title('Potassium Content Distribution')
ax.set_xlabel('Potassium Level')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# User input for prediction
st.sidebar.title("Input Soil & Environmental Conditions")
nitrogen = st.sidebar.slider("Nitrogen Content (N)", 0, 140)
phosphorus = st.sidebar.slider("Phosphorus Content (P)", 5, 145)
potassium = st.sidebar.slider("Potassium Content (K)", 5, 205)
temperature = st.sidebar.slider("Temperature (°C)", 8.0, 45.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0)
ph = st.sidebar.slider("Soil pH", 3.5, 10.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0)

# Create a dataframe from user input
user_input = pd.DataFrame({
    'N': [nitrogen],
    'P': [phosphorus],
    'K': [potassium],
    'temperature': [temperature],
    'humidity': [humidity],
    'ph': [ph],
    'rainfall': [rainfall]
})


st.markdown("## User Input Conditions:")
st.write(user_input)

# Make predictions
prediction = rf_model.predict(user_input)
st.subheader("Recommended Crop: ")
st.markdown(f"<h2 style='color:green;'>{prediction[0]}</h2>", unsafe_allow_html=True)