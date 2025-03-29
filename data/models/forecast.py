import os
import pandas as pd
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Define base directory relative to this script's location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# Construct file paths
MEASUREMENT_FILE = os.path.join(BASE_DIR, "data/raw/measurement_data.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "predictions/predictions_task_2.json")

# Forecast configuration for each station
FORECAST_CONFIG = {
    "206": {"pollutant": "SO2",   "start": "2023-07-01 00:00:00", "end": "2023-07-31 23:00:00"},
    "211": {"pollutant": "NO2",   "start": "2023-08-01 00:00:00", "end": "2023-08-31 23:00:00"},
    "217": {"pollutant": "O3",    "start": "2023-09-01 00:00:00", "end": "2023-09-30 23:00:00"},
    "219": {"pollutant": "CO",    "start": "2023-10-01 00:00:00", "end": "2023-10-31 23:00:00"},
    "225": {"pollutant": "PM10",  "start": "2023-11-01 00:00:00", "end": "2023-11-30 23:00:00"},
    "228": {"pollutant": "PM2.5", "start": "2023-12-01 00:00:00", "end": "2023-12-31 23:00:00"}
}

def preprocess_measurement_data(df):
    """
    Preprocess and clean measurement data.
    - Drops duplicates and rows with missing Measurement date or pollutant values.
    - Converts pollutant columns and 'Station code' to numeric types.
    - Ensures Measurement date is datetime.
    """
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=["Measurement date"])
    
    # List of pollutant columns to clean
    pollutant_cols = ["SO2", "NO2", "O3", "CO", "PM10", "PM2.5"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=pollutant_cols, inplace=True)
    
    # Ensure 'Station code' is numeric and drop rows with invalid values
    if "Station code" in df.columns:
        df["Station code"] = pd.to_numeric(df["Station code"], errors="coerce")
        df.dropna(subset=["Station code"], inplace=True)
    
    # Ensure Measurement date is a datetime object
    if not pd.api.types.is_datetime64_any_dtype(df["Measurement date"]):
        df["Measurement date"] = pd.to_datetime(df["Measurement date"], errors="coerce")
    
    return df

def feature_engineering(df):
    """
    Create additional time-based features for forecasting:
    - hour: Hour of day (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - month: Month (1-12)
    """
    df = df.copy()
    df["hour"] = df["Measurement date"].dt.hour
    df["day_of_week"] = df["Measurement date"].dt.dayofweek
    df["month"] = df["Measurement date"].dt.month
    return df

def forecast_station_improved(df, station_code, pollutant, forecast_period):
    # Filter historical data for the given station
    df_station = df[df["Station code"] == int(station_code)].copy()
    if df_station.empty or pollutant not in df_station.columns:
        # If no data is available, return a forecast of zeros
        return {ts: 0.0 for ts in forecast_period}
    
    # Apply feature engineering to create time-based features
    df_station = feature_engineering(df_station)
    
    # Define features and target
    features = ["hour", "day_of_week", "month"]
    X = df_station[features]
    y = df_station[pollutant]
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a linear regression model on historical data
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Prepare forecast features for each timestamp in forecast_period
    forecast_timestamps = pd.to_datetime(forecast_period)
    forecast_df = pd.DataFrame({"Measurement date": forecast_timestamps})
    forecast_df = feature_engineering(forecast_df)
    X_forecast = forecast_df[features]
    X_forecast_scaled = scaler.transform(X_forecast)
    
    # Predict pollutant values using the trained model
    predictions = model.predict(X_forecast_scaled)
    # Clip negative predictions to 0 (assuming negative pollutant values are unrealistic)
    predictions = np.clip(predictions, 0, None)
    
    # Build forecast dictionary with timestamp strings and predicted values rounded to 2 decimals
    forecast = {ts: round(float(pred), 2) for ts, pred in zip(forecast_period, predictions)}
    return forecast

def main():
    # Load measurement data with proper date parsing
    try:
        df_measure = pd.read_csv(MEASUREMENT_FILE, parse_dates=["Measurement date"])
    except Exception as e:
        print("Error loading measurement data:", e)
        return
    
    # Preprocess the measurement data
    df_measure = preprocess_measurement_data(df_measure)
    
    result = {"target": {}}
    
    # Loop through each forecast configuration
    for station, config in FORECAST_CONFIG.items():
        pollutant = config["pollutant"]
        start = config["start"]
        end = config["end"]
        # Generate forecast timestamps with hourly frequency
        forecast_period = pd.date_range(start=start, end=end, freq="H").strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Get the forecast for this station and pollutant using the improved model
        station_forecast = forecast_station_improved(df_measure, station, pollutant, forecast_period)
        result["target"][station] = station_forecast
    
    # Write predictions to predictions_task_2.json
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print("Forecast predictions for Task 2 have been written to", OUTPUT_FILE)

if __name__ == "__main__":
    main()