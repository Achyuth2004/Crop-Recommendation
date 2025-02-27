# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.preprocessing import LabelEncoder

# # def load_models():
# #     """Load the trained models and encoders."""
# #     classifier = pickle.load(open('FERTILIZER-RECOMMENDATION\classifier.pkl', 'rb'))
# #     encoder = pickle.load(open('fertilizer.pkl', 'rb'))
# #     return classifier, encoder
# import pickle

# def load_models():
#     """Load the trained models and encoders."""
#     with open('classifier.pkl', 'rb') as classifier_file:
#         classifier = pickle.load(classifier_file)
    
#     with open('fertilizer.pkl', 'rb') as encoder_file:
#         encoder = pickle.load(encoder_file)
    
#     return classifier, encoder




# def main():
#     st.title("Fertilizer Recommendation System")
#     st.write("This system recommends the best fertilizer for a given crop and soil condition.")

#     # Sidebar input for user parameters
#     st.sidebar.header("Input Parameters")
#     temperature = st.sidebar.slider("Temperature (°C)", min_value=10, max_value=50, value=25)
#     humidity = st.sidebar.slider("Humidity (%)", min_value=10, max_value=100, value=50)
#     moisture = st.sidebar.slider("Moisture (%)", min_value=0, max_value=100, value=30)
#     soil_type = st.sidebar.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
#     crop_type = st.sidebar.selectbox("Crop Type", ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Barley"])
#     nitrogen = st.sidebar.slider("Nitrogen Content (N)", min_value=0, max_value=100, value=50)
#     potassium = st.sidebar.slider("Potassium Content (K)", min_value=0, max_value=100, value=50)
#     phosphorus = st.sidebar.slider("Phosphorus Content (P)", min_value=0, max_value=100, value=50)

#     # Load models and encoders
#     classifier, encoder = load_models()

#     # Encode categorical inputs
#     soil_encoder = LabelEncoder().fit(["Sandy", "Loamy", "Black", "Red", "Clayey"])
#     crop_encoder = LabelEncoder().fit(["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Barley"])

#     encoded_soil_type = soil_encoder.transform([soil_type])[0]
#     encoded_crop_type = crop_encoder.transform([crop_type])[0]

#     # Prepare the input array
#     input_data = np.array([[temperature, humidity, moisture, encoded_soil_type, encoded_crop_type, nitrogen, potassium, phosphorus]])

#     # Predict the fertilizer
#     if st.button("Recommend Fertilizer"):
#         prediction = classifier.predict(input_data)[0]
#         fertilizer_name = encoder.classes_[prediction]
#         st.success(f"Recommended Fertilizer: {fertilizer_name}")

#     # Display additional information
#     st.write("### About the Data")
#     st.write("The model uses parameters like temperature, humidity, soil type, crop type, and nutrient levels to predict the best fertilizer for optimal crop yield.")

# if __name__ == "__main__":
#     main()




import os
import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Function to load the trained models and encoders
def load_models():
    """Load the trained classifier and encoder."""
    base_path = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of the script
    classifier_path = os.path.join(base_path, 'classifier.pkl')
    encoder_path = os.path.join(base_path, 'fertilizer.pkl')

    # Check if files exist before loading
    if not os.path.exists(classifier_path):
        st.error(f"Error: Model file '{classifier_path}' not found. Please upload it.")
        return None, None

    if not os.path.exists(encoder_path):
        st.error(f"Error: Encoder file '{encoder_path}' not found. Please upload it.")
        return None, None

    # Load classifier model
    with open(classifier_path, 'rb') as classifier_file:
        classifier = pickle.load(classifier_file)

    # Load fertilizer encoder
    with open(encoder_path, 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)

    return classifier, encoder


# Streamlit App UI
def main():
    st.title("🌱 Fertilizer Recommendation System")
    st.write("This system recommends the best fertilizer based on crop and soil conditions.")

    # Sidebar input for user parameters
    st.sidebar.header("🔧 Input Parameters")
    temperature = st.sidebar.slider("🌡 Temperature (°C)", min_value=10, max_value=50, value=25)
    humidity = st.sidebar.slider("💧 Humidity (%)", min_value=10, max_value=100, value=50)
    moisture = st.sidebar.slider("💦 Moisture (%)", min_value=0, max_value=100, value=30)
    soil_type = st.sidebar.selectbox("🪵 Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
    crop_type = st.sidebar.selectbox("🌾 Crop Type", ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Barley"])
    nitrogen = st.sidebar.slider("🧪 Nitrogen (N)", min_value=0, max_value=100, value=50)
    potassium = st.sidebar.slider("🧪 Potassium (K)", min_value=0, max_value=100, value=50)
    phosphorus = st.sidebar.slider("🧪 Phosphorus (P)", min_value=0, max_value=100, value=50)

    # Load models
    classifier, encoder = load_models()
    if classifier is None or encoder is None:
        return  # Stop execution if models are missing

    # Encode categorical inputs
    soil_encoder = LabelEncoder().fit(["Sandy", "Loamy", "Black", "Red", "Clayey"])
    crop_encoder = LabelEncoder().fit(["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Barley"])

    encoded_soil_type = soil_encoder.transform([soil_type])[0]
    encoded_crop_type = crop_encoder.transform([crop_type])[0]

    # Prepare input data for prediction
    input_data = np.array([[temperature, humidity, moisture, encoded_soil_type, encoded_crop_type, nitrogen, potassium, phosphorus]])

    # Predict fertilizer recommendation
    if st.button("🚜 Recommend Fertilizer"):
        prediction = classifier.predict(input_data)[0]
        fertilizer_name = encoder.classes_[prediction]
        st.success(f"✅ Recommended Fertilizer: **{fertilizer_name}**")

    # Additional information
    st.write("### 📌 About the Model")
    st.write("The model analyzes soil properties, crop type, and nutrient levels to suggest the best fertilizer for maximum yield.")

# Run the app
if __name__ == "__main__":
    main()