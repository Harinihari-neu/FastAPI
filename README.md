# FAST API lab Documentation

## Overview

In this Lab, we will learn how to expose ML models as APIs using [FastAPI](https://fastapi.tiangolo.com/) and [uvicorn](https://www.uvicorn.org/).

1. **FastAPI**: FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.
2. **uvicorn**: Uvicorn is an [Asynchronous Server Gateway Interface - ASGI](https://youtu.be/vKjCkeJGbNk) web server implementation for Python. It is often used to serve FastAPI applications.

The workflow involves the following steps:
1. Training a Random Forest Classifier on Wine Dataset.
2. Implementing feature scaling using StandardScaler for improved model performance.
3. Serving the trained model as an API using FastAPI and uvicorn.

## Dataset Information

This project uses the **Wine Dataset** from scikit-learn, which contains:
- **178 samples** of wine from three different cultivars
- **13 chemical features** including alcohol content, acidity, phenols, etc.
- **3 classes** representing different wine cultivars from the same region in Italy

### Features:
1. Alcohol
2. Malic Acid
3. Ash
4. Alcalinity of Ash
5. Magnesium
6. Total Phenols
7. Flavanoids
8. Nonflavanoid Phenols
9. Proanthocyanins
10. Color Intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

## Setting up the Lab

1. Create a virtual environment (e.g. **lab_02**).
   ```bash
   python -m venv lab_02
   ```

2. Activate the environment:
   ```bash
   # Windows
   lab_02\Scripts\activate
   
   # macOS/Linux
   source lab_02/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install fastapi[all] scikit-learn pandas
   ```

### Project Structure

```
FastAPI
├── model/
│   ├── wine_model.pkl      # Trained Random Forest model
│   └── scaler.pkl          # Fitted StandardScaler
├── src/
│   ├── __init__.py
│   ├── data.py             # Data loading and preprocessing
│   ├── main.py             # FastAPI application
│   ├── predict.py          # Prediction logic
│   └── train.py            # Model training script
├── README.md
└── requirements.txt
```

Note:
- **fastapi[all]** will install optional additional dependencies for FastAPI which includes **uvicorn**.
- The **model/** directory will be populated after running the training script.

## Running the Lab

### Step 1: Train the Model

First, navigate to the **src/** folder:
```bash
cd src
```

Train the Random Forest Classifier:
```bash
python train.py
```

**Expected Output:**
```
Loading Wine dataset...
Dataset loaded: 178 samples, 13 features
Features: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium...

Splitting and scaling data...
Training Random Forest Classifier...
Evaluating model...
==================================================
Model Evaluation Results
==================================================
Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support

     Class 0       1.00      1.00      1.00        15
     Class 1       1.00      1.00      1.00        18
     Class 2       1.00      1.00      1.00        12

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45

==================================================

Saving model and scaler...
✓ Model saved to ../model/wine_model.pkl
✓ Scaler saved to ../model/scaler.pkl

✓ Training complete! You can now run the FastAPI server.
```

### Step 2: Start the API Server

From the **src/** directory, run:
```bash
uvicorn main:app --reload
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 3: Test the API

Navigate to the interactive API documentation at:
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 
- [http://localhost:8000/docs](http://localhost:8000/docs)

You can test the endpoints by:
1. Clicking on the dropdown for any endpoint
2. Clicking **"Try it out"**
3. Filling in the Request body
4. Clicking **"Execute"**

You can also use tools like [Postman](https://www.postman.com/) for API testing.

## API Endpoints

### 1. Root Endpoint
- **Method**: GET
- **Path**: `/`
- **Description**: Returns API information and available endpoints
- **Response**:
```json
{
  "message": "Wine Quality Prediction API",
  "version": "1.0.0",
  "endpoints": {
    "POST /predict": "Predict wine class from chemical properties",
    "GET /docs": "Interactive API documentation",
    "GET /health": "Health check endpoint"
  }
}
```

### 2. Health Check
- **Method**: GET
- **Path**: `/health`
- **Description**: Verifies API and model status
- **Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Wine Class Information
- **Method**: GET
- **Path**: `/wine-classes`
- **Description**: Returns information about wine classes
- **Response**:
```json
{
  "classes": {
    "0": "Class 0 (Cultivar 1)",
    "1": "Class 1 (Cultivar 2)",
    "2": "Class 2 (Cultivar 3)"
  },
  "description": "Three cultivars of wine from the same region in Italy"
}
```

### 4. Predict Wine Class
- **Method**: POST
- **Path**: `/predict`
- **Description**: Predicts wine class based on chemical properties
- **Request Body**:
```json
{
  "alcohol": 13.2,
  "malic_acid": 2.77,
  "ash": 2.51,
  "alcalinity_of_ash": 18.5,
  "magnesium": 96.0,
  "total_phenols": 2.45,
  "flavanoids": 2.53,
  "nonflavanoid_phenols": 0.39,
  "proanthocyanins": 1.52,
  "color_intensity": 4.6,
  "hue": 1.03,
  "od280_od315": 3.0,
  "proline": 770.0
}
```

- **Response**:
```json
{
  "wine_class": 0,
  "class_name": "Class 0 (Cultivar 1)",
  "probabilities": {
    "Class 0 (Cultivar 1)": 0.9567,
    "Class 1 (Cultivar 2)": 0.0312,
    "Class 2 (Cultivar 3)": 0.0121
  }
}
```

## FastAPI Implementation Details

### FastAPI Syntax

The instance of FastAPI class is defined as:
```python
app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for predicting wine class based on chemical properties",
    version="1.0.0"
)
```

When you run a FastAPI application, you pass this app instance to the ASGI server uvicorn:
```bash
uvicorn main:app --reload
```

In this command:
- **main** is the name of the Python file containing your app instance (without the .py extension)
- **app** is the name of the instance itself
- **--reload** flag tells uvicorn to restart the server whenever code changes are detected (useful for development, not for production)

### Route Handlers

All functions that serve as API endpoints are prefixed with decorators like **@app.get()** or **@app.post()**:

```python
@app.get("/")
async def root():
    return {"message": "Wine Quality Prediction API"}

@app.post("/predict", response_model=WinePrediction)
async def predict_wine(wine_data: WineData):
    # Prediction logic
    pass
```

**Decorator Components:**
1. **Decorator (@)**: Used to associate a function with a particular HTTP method and path
2. **App Instance (app)**: The FastAPI application instance
3. **HTTP Method (get, post, etc.)**: Specifies the type of HTTP request (GET for retrieving, POST for sending data)
4. **Path/Endpoint**: The URL path where the API is accessible

For more details on HTTP methods, refer to [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods).

### Async Functions

Using **async** in FastAPI allows for non-blocking operations, enabling the server to handle other requests while waiting for I/O tasks (like model loading or database queries) to complete. This improves concurrency and resource utilization.

### Data Models with Pydantic

#### 1. WineData Class (Input Model)

```python
class WineData(BaseModel):
    alcohol: float = Field(..., description="Alcohol content", ge=0, le=20)
    malic_acid: float = Field(..., description="Malic acid content", ge=0)
    ash: float = Field(..., description="Ash content", ge=0)
    # ... other fields
```

The **WineData** class is a [Pydantic model](https://docs.pydantic.dev/latest/concepts/models/) that defines the expected structure of the request body. FastAPI performs:

- **Request Body Reading**: Reads the request body as JSON
- **Data Conversion**: Converts data to proper Python types (e.g., string "5.1" to float 5.1)
- **Data Validation**: Validates all required fields and constraints. Returns **422 Unprocessable Entity** if validation fails

**Field Validations:**
- `ge=0`: Greater than or equal to 0
- `le=20`: Less than or equal to 20
- `...`: Field is required

#### 2. WinePrediction Class (Output Model)

```python
class WinePrediction(BaseModel):
    wine_class: int = Field(..., description="Predicted wine class (0, 1, or 2)")
    class_name: str = Field(..., description="Human-readable class name")
    probabilities: dict = Field(..., description="Prediction probabilities")
```

When you specify **response_model=WinePrediction**, FastAPI will:
- **Serialize Output**: Convert output data to JSON format
- **Document API**: Include the model in API documentation so consumers know what to expect

### Error Handling

Error handling uses the **HTTPException** class:

```python
from fastapi import HTTPException

@app.post("/predict")
async def predict_wine(wine_data: WineData):
    try:
        # Prediction logic
        pass
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please train the model first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
```

**HTTPException Components:**
- **status_code**: HTTP status code (404 for Not Found, 422 for Validation Error, 500 for Server Error, etc.)
- **detail**: Description of the error (returned as JSON)

**Example Error Response:**
```json
{
  "detail": "Model not found. Please train the model first by running 'python train.py'"
}
```

For more information on error handling, refer to [FastAPI Error Handling Documentation](https://fastapi.tiangolo.com/tutorial/handling-errors/).

## Model Architecture

### Algorithm: Random Forest Classifier

**Parameters:**
- `n_estimators=100`: 100 decision trees in the forest
- `max_depth=5`: Maximum depth of each tree
- `random_state=42`: For reproducibility
- `n_jobs=-1`: Use all available CPU cores

### Preprocessing: StandardScaler

Features are scaled to have:
- Mean = 0
- Standard deviation = 1

This improves model performance and convergence.

### Train-Test Split

- **Training set**: 75% of data (stratified by class)
- **Test set**: 25% of data
- **random_state=42**: Ensures reproducible splits

## Test Cases

### Test Case 1: Class 0 (Cultivar 1)
```json
{
  "alcohol": 13.2,
  "malic_acid": 2.77,
  "ash": 2.51,
  "alcalinity_of_ash": 18.5,
  "magnesium": 96.0,
  "total_phenols": 2.45,
  "flavanoids": 2.53,
  "nonflavanoid_phenols": 0.39,
  "proanthocyanins": 1.52,
  "color_intensity": 4.6,
  "hue": 1.03,
  "od280_od315": 3.0,
  "proline": 770.0
}
```

### Test Case 2: Class 1 (Cultivar 2)
```json
{
  "alcohol": 12.5,
  "malic_acid": 1.73,
  "ash": 2.12,
  "alcalinity_of_ash": 19.0,
  "magnesium": 80.0,
  "total_phenols": 1.65,
  "flavanoids": 2.03,
  "nonflavanoid_phenols": 0.37,
  "proanthocyanins": 1.63,
  "color_intensity": 3.4,
  "hue": 1.0,
  "od280_od315": 3.17,
  "proline": 510.0
}
```

### Test Case 3: Class 2 (Cultivar 3)
```json
{
  "alcohol": 13.5,
  "malic_acid": 3.0,
  "ash": 2.6,
  "alcalinity_of_ash": 23.0,
  "magnesium": 105.0,
  "total_phenols": 1.77,
  "flavanoids": 1.2,
  "nonflavanoid_phenols": 0.6,
  "proanthocyanins": 1.05,
  "color_intensity": 5.8,
  "hue": 0.88,
  "od280_od315": 1.93,
  "proline": 1285.0
}
```

### Invalid Input Test (Validation Error)
```json
{
  "alcohol": "not_a_number",
  "malic_acid": 2.77,
  "ash": 2.51,
  "alcalinity_of_ash": 18.5,
  "magnesium": 96.0,
  "total_phenols": 2.45,
  "flavanoids": 2.53,
  "nonflavanoid_phenols": 0.39,
  "proanthocyanins": 1.52,
  "color_intensity": 4.6,
  "hue": 1.03,
  "od280_od315": 3.0,
  "proline": 770.0
}
```
**Expected Response**: 422 Unprocessable Entity with validation details

## Key Differences from Base Implementation

1.  **Different Dataset**: Wine dataset (178 samples, 13 features) instead of Iris (150 samples, 4 features)
2.  **Different Algorithm**: Random Forest Classifier instead of Decision Tree
3.  **Feature Scaling**: Implemented StandardScaler for improved model performance
4.  **Enhanced API**: Added multiple endpoints (health check, wine class info)
5.  **Probability Predictions**: Returns confidence scores for all classes
6.  **Comprehensive Validation**: Field-level validation with constraints
7.  **Model Evaluation**: Detailed classification report and accuracy metrics
8.  **Better Documentation**: Comprehensive docstrings and API descriptions

## Model Performance

- **Accuracy**: 100% on test set (45 samples)
- **Precision, Recall, F1-Score**: 1.00 for all classes
- **Perfect classification** on all three wine cultivars

## Troubleshooting

### Issue 1: Module Not Found
```bash
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: Ensure virtual environment is activated and packages are installed:
```bash
pip install fastapi[all] scikit-learn pandas
```

### Issue 2: Model Not Found
```bash
FileNotFoundError: [Errno 2] No such file or directory: '../model/wine_model.pkl'
```
**Solution**: Train the model first:
```bash
cd src
python train.py
```

### Issue 3: Port Already in Use
```bash
ERROR: [Errno 48] error while attempting to bind on address ('127.0.0.1', 8000)
```
**Solution**: Use a different port:
```bash
uvicorn main:app --reload --port 8001
```

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Scikit-learn Wine Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)
- [HTTP Methods - MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)

