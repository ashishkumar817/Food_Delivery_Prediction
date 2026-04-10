from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import os

app = FastAPI(title="Food Delivery Prediction API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = 'model.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

class PredictionInput(BaseModel):
    distance_km: float
    preparation_time_min: float
    courier_experience_yrs: float
    weather: str  # Foggy, Rainy, Snowy, Windy, Clear
    traffic_level: str # Low, Medium, High
    time_of_day: str # Evening, Morning, Night, Afternoon
    vehicle_type: str # Car, Scooter, Bike

@app.get("/")
def read_root():
    return {"message": "Food Delivery Prediction API is running"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")

    try:
        # Preprocessing: Convert input to the format used during training (One-Hot Encoding)
        # Features used in training:
        # ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 
        #  'Weather_Foggy', 'Weather_Rainy', 'Weather_Snowy', 'Weather_Windy', 
        #  'Traffic_Level_Low', 'Traffic_Level_Medium', 
        #  'Time_of_Day_Evening', 'Time_of_Day_Morning', 'Time_of_Day_Night', 
        #  'Vehicle_Type_Car', 'Vehicle_Type_Scooter']
        
        weather = input_data.weather.lower()
        traffic = input_data.traffic_level.lower()
        time_day = input_data.time_of_day.lower()
        vehicle = input_data.vehicle_type.lower()

        data_dict = {
            'Distance_km': [input_data.distance_km],
            'Preparation_Time_min': [input_data.preparation_time_min],
            'Courier_Experience_yrs': [input_data.courier_experience_yrs],
            'Weather_Foggy': [1 if weather == "foggy" else 0],
            'Weather_Rainy': [1 if weather == "rainy" else 0],
            'Weather_Snowy': [1 if weather == "snowy" else 0],
            'Weather_Windy': [1 if weather == "windy" else 0],
            'Traffic_Level_Low': [1 if traffic == "low" else 0],
            'Traffic_Level_Medium': [1 if traffic == "medium" else 0],
            'Time_of_Day_Evening': [1 if time_day == "evening" else 0],
            'Time_of_Day_Morning': [1 if time_day == "morning" else 0],
            'Time_of_Day_Night': [1 if time_day == "night" else 0],
            'Vehicle_Type_Car': [1 if vehicle == "car" else 0],
            'Vehicle_Type_Scooter': [1 if vehicle == "scooter" else 0]
        }

        X_new = pd.DataFrame(data_dict)
        
        # Predict
        prediction = model.predict(X_new)[0]
        
        # Status calculation
        status = "On-time" if prediction <= 30 else "Late"
        
        return {
            "predicted_delivery_time": round(float(prediction), 2),
            "delivery_status": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
