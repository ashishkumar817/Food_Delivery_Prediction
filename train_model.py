import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_and_save_model():
    # Load dataset
    csv_path = 'Food_Delivery_Times.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Data Cleaning (replication of notebook logic)
    # Categorical -> mode
    for col in ['Weather', 'Traffic_Level', 'Time_of_Day']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Numerical -> median
    if 'Courier_Experience_yrs' in df.columns:
        df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median())

    # Encoding (Manual replication of the notebook's one-hot encoding structure)
    # The notebook drops 'Order_ID' and focuses on specific features.
    
    encoded_df = pd.get_dummies(df, columns=['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type'])
    
    # Selecting the exact features used in the notebook for regression
    # Features required: Distance_km, Preparation_Time_min, Courier_Experience_yrs, 
    # Weather_Foggy, Weather_Rainy, Weather_Snowy, Weather_Windy, 
    # Traffic_Level_Low, Traffic_Level_Medium, 
    # Time_of_Day_Evening, Time_of_Day_Morning, Time_of_Day_Night, 
    # Vehicle_Type_Car, Vehicle_Type_Scooter

    required_features = [
        'Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs',
        'Weather_Foggy', 'Weather_Rainy', 'Weather_Snowy', 'Weather_Windy',
        'Traffic_Level_Low', 'Traffic_Level_Medium',
        'Time_of_Day_Evening', 'Time_of_Day_Morning', 'Time_of_Day_Night',
        'Vehicle_Type_Car', 'Vehicle_Type_Scooter'
    ]

    # Ensure all columns exist (even if 0 if some category wasn't in sample)
    for col in required_features:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

    X = encoded_df[required_features].astype(int)
    y = df['Delivery_Time_min']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved as model.pkl")
    print(f"Features used: {required_features}")

if __name__ == "__main__":
    train_and_save_model()
