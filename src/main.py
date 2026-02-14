from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import sys
sys.path.append('..')
from predict import predict_wine_class

# Initialize FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for predicting wine class based on chemical properties",
    version="1.0.0"
)

# Define input data model
class WineData(BaseModel):
    """
    Input features for wine classification.
    All features are chemical properties of wine samples.
    """
    alcohol: float = Field(..., description="Alcohol content", ge=0, le=20)
    malic_acid: float = Field(..., description="Malic acid content", ge=0)
    ash: float = Field(..., description="Ash content", ge=0)
    alcalinity_of_ash: float = Field(..., description="Alcalinity of ash", ge=0)
    magnesium: float = Field(..., description="Magnesium content", ge=0)
    total_phenols: float = Field(..., description="Total phenols", ge=0)
    flavanoids: float = Field(..., description="Flavanoids content", ge=0)
    nonflavanoid_phenols: float = Field(..., description="Nonflavanoid phenols", ge=0)
    proanthocyanins: float = Field(..., description="Proanthocyanins", ge=0)
    color_intensity: float = Field(..., description="Color intensity", ge=0)
    hue: float = Field(..., description="Hue", ge=0)
    od280_od315: float = Field(..., description="OD280/OD315 of diluted wines", ge=0)
    proline: float = Field(..., description="Proline content", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }

# Define output data model
class WinePrediction(BaseModel):
    """
    Prediction response containing wine class and confidence scores.
    """
    wine_class: int = Field(..., description="Predicted wine class (0, 1, or 2)")
    class_name: str = Field(..., description="Human-readable class name")
    probabilities: dict = Field(..., description="Prediction probabilities for each class")

# Wine class mapping
WINE_CLASSES = {
    0: "Class 0 (Cultivar 1)",
    1: "Class 1 (Cultivar 2)", 
    2: "Class 2 (Cultivar 3)"
}

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Wine Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict wine class from chemical properties",
            "GET /docs": "Interactive API documentation",
            "GET /health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    try:
        from predict import load_model_and_scaler
        load_model_and_scaler()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/predict", response_model=WinePrediction)
async def predict_wine(wine_data: WineData):
    """
    Predict wine class based on chemical properties.
    
    Args:
        wine_data (WineData): Wine features including alcohol, acids, phenols, etc.
    
    Returns:
        WinePrediction: Predicted class and probability scores.
    """
    try:
        # Convert input data to list
        features = [
            wine_data.alcohol,
            wine_data.malic_acid,
            wine_data.ash,
            wine_data.alcalinity_of_ash,
            wine_data.magnesium,
            wine_data.total_phenols,
            wine_data.flavanoids,
            wine_data.nonflavanoid_phenols,
            wine_data.proanthocyanins,
            wine_data.color_intensity,
            wine_data.hue,
            wine_data.od280_od315,
            wine_data.proline
        ]
        
        # Get prediction
        prediction, probabilities = predict_wine_class(features)
        
        # Format response
        response = {
            "wine_class": prediction,
            "class_name": WINE_CLASSES[prediction],
            "probabilities": {
                WINE_CLASSES[i]: round(prob, 4) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        return response
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please train the model first by running 'python train.py'"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/wine-classes")
async def get_wine_classes():
    """
    Get information about wine classes.
    """
    return {
        "classes": WINE_CLASSES,
        "description": "Three cultivars of wine from the same region in Italy"
    }