import sys
print("Script started!", file=sys.stderr)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data import load_data, split_and_scale_data

def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier for wine classification.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    
    Returns:
        model: Trained Random Forest model.
    """
    # Using Random Forest instead of Decision Tree for variety
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained classifier.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test target values.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Model Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Class 0', 'Class 1', 'Class 2']))
    print(f"{'='*50}\n")

def save_model_and_scaler(model, scaler):
    """
    Save the trained model and scaler to disk.
    
    Args:
        model: Trained classifier.
        scaler: Fitted StandardScaler.
    """
    joblib.dump(model, "../model/wine_model.pkl")
    joblib.dump(scaler, "../model/scaler.pkl")
    print("✓ Model saved to ../model/wine_model.pkl")
    print("✓ Scaler saved to ../model/scaler.pkl")

if __name__ == "__main__":
    print("Loading Wine dataset...")
    X, y, feature_names = load_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {', '.join(feature_names[:5])}...\n")
    
    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    print("Training Random Forest Classifier...")
    model = train_model(X_train, y_train)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    print("Saving model and scaler...")
    save_model_and_scaler(model, scaler)
    
    print("\n✓ Training complete! You can now run the FastAPI server.")