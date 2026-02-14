import joblib
import numpy as np

def load_model_and_scaler():
    """
    Load the trained model and scaler from disk.
    
    Returns:
        model: Loaded classifier.
        scaler: Loaded StandardScaler.
    """
    model = joblib.load("../model/wine_model.pkl")
    scaler = joblib.load("../model/scaler.pkl")
    return model, scaler

def predict_wine_class(features):
    """
    Predict the wine class for the given features.
    
    Args:
        features (list or numpy.ndarray): Wine chemical properties (13 features).
    
    Returns:
        prediction (int): Predicted wine class (0, 1, or 2).
        probability (list): Prediction probabilities for each class.
    """
    model, scaler = load_model_and_scaler()
    
    # Ensure features is a 2D array
    if isinstance(features, list):
        features = np.array(features).reshape(1, -1)
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return int(prediction), probability.tolist()