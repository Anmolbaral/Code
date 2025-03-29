import os
import pandas as pd
import json

# Define base directory relative to this script's location (three levels up)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# Construct file paths
INSTRUMENT_FILE = os.path.join(BASE_DIR, "data/raw/instrument_data.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "predictions/predictions_task_3.json")

def preprocess_instrument_data(df):
    """
    Preprocess and clean instrument data:
    - Drop duplicate rows.
    - Remove rows missing critical columns: Measurement date, Station code, Instrument status.
    - Convert 'Measurement date' to datetime.
    - Strip whitespace from Instrument status if it's string, then convert 'Instrument status' and 'Station code' to numeric.
    - Create an 'hour_ts' column that floors Measurement date to the hour.
    """
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=["Measurement date", "Station code", "Instrument status"])
    
    # Ensure Measurement date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Measurement date"]):
        df["Measurement date"] = pd.to_datetime(df["Measurement date"], errors="coerce")
    
    # Strip whitespace if Instrument status is string, then convert columns to numeric
    if df["Instrument status"].dtype == object:
        df["Instrument status"] = df["Instrument status"].str.strip()
    df["Instrument status"] = pd.to_numeric(df["Instrument status"], errors="coerce")
    
    df["Station code"] = pd.to_numeric(df["Station code"], errors="coerce")
    df = df.dropna(subset=["Station code", "Instrument status"])
    
    # Create a column with the Measurement date floored to the hour
    df["hour_ts"] = df["Measurement date"].dt.floor("H")
    
    return df

def detect_anomalies_for_station(df, station_code, start, end):
    """
    For a given station and time period, count the number of anomaly records (where Instrument status != 0)
    for each hour in the period.
    """
    # Filter for the specific station
    df_station = df[df["Station code"] == int(station_code)].copy()
    if df_station.empty:
        print(f"No records found for station {station_code}.")
        return {}
    
    # Convert start and end to datetime and filter data within the period
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    df_station = df_station[(df_station["Measurement date"] >= start_ts) & 
                            (df_station["Measurement date"] <= end_ts)]
    
    print(f"Station {station_code}: {len(df_station)} records between {start} and {end}.")
    
    # Select records where Instrument status is not 0 (anomaly condition)
    df_anomalies = df_station[df_station["Instrument status"] != 0]
    print(f"Station {station_code}: Found {len(df_anomalies)} anomaly records (status != 0).")
    if not df_anomalies.empty:
        print(f"Sample anomalies for station {station_code}:\n", df_anomalies.head())
    
    # Create the full hourly range for the period
    hourly_range = pd.date_range(start=start, end=end, freq="H")
    
    # Group anomalies by hour_ts and count
    anomaly_group = df_anomalies.groupby("hour_ts").size().to_dict()
    
    # Build final dictionary using the full hourly range; if no anomalies in an hour, count is 0.
    hourly_anomalies = {}
    for ts in hourly_range:
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        count = anomaly_group.get(ts, 0)
        hourly_anomalies[ts_str] = int(count)
    
    return hourly_anomalies

def main():
    # Load instrument data with date parsing
    try:
        df_instrument = pd.read_csv(INSTRUMENT_FILE, parse_dates=["Measurement date"])
    except Exception as e:
        print("Error loading instrument data:", e)
        return

    # Preprocess instrument data
    df_instrument = preprocess_instrument_data(df_instrument)
    
    # Debug: Print overall date range in the instrument data
    overall_min = df_instrument["Measurement date"].min()
    overall_max = df_instrument["Measurement date"].max()
    print(f"Overall instrument data date range: {overall_min} to {overall_max}")
    
    result = {"target": {}}
    
    # Define anomaly detection configuration for each station and its period
    ANOMALY_CONFIG = {
        "205": {"start": "2023-11-01 00:00:00", "end": "2023-11-30 23:00:00"},
        "209": {"start": "2023-09-01 00:00:00", "end": "2023-09-30 23:00:00"},
        "223": {"start": "2023-07-01 00:00:00", "end": "2023-07-31 23:00:00"},
        "224": {"start": "2023-10-01 00:00:00", "end": "2023-10-31 23:00:00"},
        "226": {"start": "2023-08-01 00:00:00", "end": "2023-08-31 23:00:00"},
        "227": {"start": "2023-12-01 00:00:00", "end": "2023-12-31 23:00:00"}
    }
    
    # Loop through each configuration entry and detect anomalies
    for station, config in ANOMALY_CONFIG.items():
        start = config["start"]
        end = config["end"]
        anomalies = detect_anomalies_for_station(df_instrument, station, start, end)
        result["target"][station] = anomalies
    
    # Write the anomaly detection results to the output JSON file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print("Anomaly detection results for Task 3 have been written to", OUTPUT_FILE)

if __name__ == "__main__":
    main()