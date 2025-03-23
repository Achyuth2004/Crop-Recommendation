from django.shortcuts import render
import pickle
import numpy as np

def home(request):
    return render(request, 'index.html')

def getPredictions(N, P, K, temp, humidity, ph, rain, soil_type):
    model = pickle.load(open('SVM_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    # Preprocess input data
    data = np.array([[N, P, K, temp, humidity, ph, rain, soil_type]])
    data = scaler.transform(data)

    # Make predictions and get probability
    prediction = model.predict(data)
    prediction_prob = model.predict_proba(data)

    return prediction, prediction_prob

def result(request):
    # Accessing the form values properly without extra spaces
    try:
        N = float(request.GET.get('N', 0))  # Use .get() to avoid MultiValueDictKeyError
        P = float(request.GET.get('P', 0))
        K = float(request.GET.get('K', 0))
        temp = float(request.GET.get('TEMPERATURE', 0))  # Corrected from 'TEMPARATURE'
        humidity = float(request.GET.get('HUMIDITY', 0))
        ph = float(request.GET.get('PH', 0))
        rain = float(request.GET.get('RAIN', 0))
        soil_type = float(request.GET.get('soil_type', 0))

        # Get the prediction and its probability
        prediction, prediction_prob = getPredictions(N, P, K, temp, humidity, ph, rain, soil_type)

        # Extract the class and its probability
        predicted_class = prediction[0]
        confidence = np.max(prediction_prob) * 100  # Convert probability to percentage

        # Pass both the result and confidence to the template
        return render(request, 'result.html', {'result': predicted_class, 'confidence': confidence})

    except ValueError as e:
        return render(request, 'error.html', {'error_message': 'Invalid input. Please check your values.'})

