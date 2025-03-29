import os
import pandas as pd
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
INSTRUMENT_FILE = os.path.join(BASE_DIR, "data/raw/instrument_data.csv")
MEASUREMENT_FILE = os.path.join(BASE_DIR, "data/raw/measurement_data.csv")
POLLUTANT_FILE = os.path.join(BASE_DIR, "data/raw/pollutant_data.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "predictions/predictions_task_3.json")

ANOMALY_CONFIG = {
    "205": {"pollutant": "SO2", "start": "2023-11-01 00:00:00", "end": "2023-11-30 23:00:00"},
    "209": {"pollutant": "NO2", "start": "2023-09-01 00:00:00", "end": "2023-09-30 23:00:00"},
    "223": {"pollutant": "O3", "start": "2023-07-01 00:00:00", "end": "2023-07-31 23:00:00"},
    "224": {"pollutant": "CO", "start": "2023-10-01 00:00:00", "end": "2023-10-31 23:00:00"},
    "226": {"pollutant": "PM10", "start": "2023-08-01 00:00:00", "end": "2023-08-31 23:00:00"},
    "227": {"pollutant": "PM2.5", "start": "2023-12-01 00:00:00", "end": "2023-12-31 23:00:00"}
}

def load_pollutant_mapping():
    """Load pollutant name to item code mapping."""
    try:
        df = pd.read_csv(POLLUTANT_FILE)
        return df.set_index('Item name')['Item code'].to_dict()
    except Exception as e:
        print(f"Error loading pollutant data: {str(e)}")
        return {}

def preprocess_instrument_data(df, pollutant_map):
    """Clean and enhance instrument data with pollution-specific features."""
    # Convert critical columns
    df["Measurement date"] = pd.to_datetime(df["Measurement date"], errors="coerce")
    df["Station code"] = pd.to_numeric(df["Station code"], errors="coerce")
    df["Item code"] = pd.to_numeric(df["Item code"], errors="coerce")
    df["Instrument status"] = pd.to_numeric(df["Instrument status"], errors="coerce")
    
    # Add pollution name mapping using reverse mapping
    reverse_pollutant_map = {v: k for k, v in pollutant_map.items()}
    df["Pollutant"] = df["Item code"].map(reverse_pollutant_map)
    
    # Drop rows with missing critical fields and duplicates
    df = df.dropna(subset=["Measurement date", "Station code", "Item code", "Instrument status"])
    df = df.drop_duplicates(subset=["Measurement date", "Station code", "Item code"])
    
    # Create hourly timestamp
    df["hour_ts"] = df["Measurement date"].dt.floor("H")
    return df

def generate_hourly_template(start, end):
    """Generate complete hourly index for the period."""
    return pd.date_range(
        start=pd.to_datetime(start).floor("H"),
        end=pd.to_datetime(end).ceil("H"),
        freq="H"
    )

def train_model(X, y):
    """Train a classification model with imputation to handle missing values."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Pipeline: impute missing values -> scale -> RandomForestClassifier
    model = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    )
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    return model

def predict_anomalies(model, features, full_index):
    """Predict anomalies for a given period.
       Returns a dictionary mapping timestamps to predicted anomaly values (non-normal statuses only).
    """
    predictions = model.predict(features)
    return {
        ts.strftime("%Y-%m-%d %H:%M:%S"): int(pred)
        for ts, pred in zip(full_index, predictions)
        if pred != 0  # Only include non-normal statuses in the output
    }

def main():
    pollutant_map = load_pollutant_mapping()
    if not pollutant_map:
        print("Critical error: Failed to load pollutant mapping")
        return

    try:
        # Load and preprocess instrument data
        instrument_df = pd.read_csv(INSTRUMENT_FILE)
        instrument_df = preprocess_instrument_data(instrument_df, pollutant_map)
        
        # Merge with measurement data if necessary for feature generation.
        # For simplicity, we assume features for training are available from instrument_df.
        # You may want to merge with measurement_df similarly to your previous approach.
        
        # For training, select features and target:
        # Here, we'll assume that 'Instrument status' is the target,
        # and we use a set of features, e.g., hour, day_of_week, and month.
        instrument_df["Measurement date"] = pd.to_datetime(instrument_df["Measurement date"])
        instrument_df["hour"] = instrument_df["Measurement date"].dt.hour
        instrument_df["day_of_week"] = instrument_df["Measurement date"].dt.dayofweek
        instrument_df["month"] = instrument_df["Measurement date"].dt.month
        
        # Define features and target; you may adjust this list based on your data.
        features = instrument_df[["hour", "day_of_week", "month"]]
        target = instrument_df["Instrument status"]
        
        # Train model with imputation to handle any NaN values in features.
        model = train_model(features, target)
        
        # Generate predictions for each station in ANOMALY_CONFIG
        result = {"target": {}}
        for station, config in ANOMALY_CONFIG.items():
            print(f"Processing {config['pollutant']} anomalies for station {station}...")
            # Generate full hourly index for the forecast period.
            full_index = generate_hourly_template(config["start"], config["end"])
            
            # Here, you need to generate 'future_features' for the prediction period.
            # This can be done by constructing a DataFrame with the necessary feature columns
            # (e.g., hour, day_of_week, month) based on the timestamps in full_index.
            future_df = pd.DataFrame({"Measurement date": full_index})
            future_df["hour"] = future_df["Measurement date"].dt.hour
            future_df["day_of_week"] = future_df["Measurement date"].dt.dayofweek
            future_df["month"] = future_df["Measurement date"].dt.month
            future_features = future_df[["hour", "day_of_week", "month"]]
            
            station_preds = predict_anomalies(model, future_features, full_index)
            result["target"][station] = station_preds
        
        # Save results, converting any non-native types to int if needed.
        with open(OUTPUT_FILE, "w") as f:
            json.dump(result, f, indent=2, default=lambda x: int(x))
            
        print(f"Successfully generated anomaly report at {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Execution failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
