import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Load the Wine dataset and return the features and target values.
    
    Returns:
        X (numpy.ndarray): The features of the Wine dataset (13 chemical properties).
        y (numpy.ndarray): The target values (wine class: 0, 1, or 2).
        feature_names (list): Names of the features.
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    return X, y, feature_names

def split_and_scale_data(X, y):
    """
    Split the data into training and testing sets, and scale the features.
    
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split and scaled dataset.
        scaler (StandardScaler): Fitted scaler for future use.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale features for better model performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler