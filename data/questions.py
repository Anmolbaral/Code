import os
import pandas as pd
import json

# Define base directory relative to this script's location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Construct file paths
MEASUREMENT_FILE = os.path.join(BASE_DIR, "data/raw/measurement_data.csv")
INSTRUMENT_FILE = os.path.join(BASE_DIR, "data/raw/instrument_data.csv")
POLLUTANT_FILE = os.path.join(BASE_DIR, "data/raw/pollutant_data.csv")

def get_season(month):
    # Map month to season code: 1: Winter, 2: Spring, 3: Summer, 4: Autumn
    if month in [12, 1, 2]:
        return "1"  # Winter
    elif month in [3, 4, 5]:
        return "2"  # Spring
    elif month in [6, 7, 8]:
        return "3"  # Summer
    else:
        return "4"  # Autumn

def preprocess_measurement_data(df):
    """
    Preprocess and clean measurement data.
    - Drops duplicates and rows with missing Measurement date or pollutant values.
    - Converts pollutant columns and 'Station code' to numeric types.
    - Ensures Measurement date is a datetime.
    """
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=["Measurement date"])
    
    pollutant_cols = ["SO2", "NO2", "O3", "CO", "PM10", "PM2.5"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=pollutant_cols, inplace=True)
    
    if "Station code" in df.columns:
        df["Station code"] = pd.to_numeric(df["Station code"], errors="coerce")
        df.dropna(subset=["Station code"], inplace=True)
    
    if not pd.api.types.is_datetime64_any_dtype(df["Measurement date"]):
        df["Measurement date"] = pd.to_datetime(df["Measurement date"], errors="coerce")
    
    return df

def feature_engineering_measurement(df):
    """
    Add additional time-based features to measurement data:
    - day: Date part of the measurement (YYYY-MM-DD)
    - month: Month extracted from Measurement date
    - season: Derived from month using get_season()
    - hour: Hour of day (0-23)
    """
    df = df.copy()
    df["day"] = df["Measurement date"].dt.date
    df["month"] = df["Measurement date"].dt.month
    df["season"] = df["month"].apply(get_season)
    df["hour"] = df["Measurement date"].dt.hour
    return df

def compute_task1():
    # Load and preprocess measurement data
    try:
        df_measure = pd.read_csv(MEASUREMENT_FILE, parse_dates=["Measurement date"])
    except Exception as e:
        print("Error loading measurement data:", e)
        return None

    df_measure = preprocess_measurement_data(df_measure)
    df_measure = feature_engineering_measurement(df_measure)
    
    # If a 'status' column exists, filter for normal measurements (status == 0)
    if "status" in df_measure.columns:
        df_normal = df_measure[df_measure["status"] == 0].copy()
    else:
        df_normal = df_measure.copy()

    # Q1: Average daily SO2 concentration across all stations.
    # First, compute daily averages per station, then take the mean of these daily values.
    daily_avg = df_normal.groupby(["Station code", "day"])["SO2"].mean()
    q1 = round(daily_avg.mean(), 5)
    
    # Q2: Average CO per season at station 209.
    df_209 = df_normal[df_normal["Station code"] == 209].copy()
    # 'season' column is already added during feature engineering
    q2_series = df_209.groupby("season")["CO"].mean().round(5)
    q2 = q2_series.to_dict()
    
    # Q3: Which hour presents the highest variability (std dev) for pollutant O3 (all stations)
    std_by_hour = df_normal.groupby("hour")["O3"].std()
    q3 = int(std_by_hour.idxmax())
    
    # Load and preprocess instrument data
    try:
        df_instrument = pd.read_csv(INSTRUMENT_FILE, parse_dates=["Measurement date"])
    except Exception as e:
        print("Error loading instrument data:", e)
        df_instrument = pd.DataFrame()
    
    if not df_instrument.empty:
        df_instrument.drop_duplicates(inplace=True)
        df_instrument = df_instrument.dropna(subset=["Measurement date", "Station code", "Instrument status"])
        df_instrument["Station code"] = pd.to_numeric(df_instrument["Station code"], errors="coerce")
        df_instrument.dropna(subset=["Station code"], inplace=True)
        df_instrument["Instrument status"] = pd.to_numeric(df_instrument["Instrument status"], errors="coerce")
    
    # Q4: Station code with most 'Abnormal data' (Instrument status code 9)
    if not df_instrument.empty and "Instrument status" in df_instrument.columns:
        df_abnormal = df_instrument[df_instrument["Instrument status"] == 9]
        q4 = int(df_abnormal["Station code"].value_counts().idxmax()) if not df_abnormal.empty else None
    else:
        q4 = None
    
    # Q5: Station code with the most 'not normal' measurements (Instrument status != 0)
    if not df_instrument.empty and "Instrument status" in df_instrument.columns:
        df_not_normal = df_instrument[df_instrument["Instrument status"] != 0]
        q5 = int(df_not_normal["Station code"].value_counts().idxmax()) if not df_not_normal.empty else None
    else:
        q5 = None
    
    # Load and preprocess pollutant data
    try:
        df_pollutant = pd.read_csv(POLLUTANT_FILE)
    except Exception as e:
        print("Error loading pollutant data:", e)
        df_pollutant = pd.DataFrame()
    
    if not df_pollutant.empty:
        df_pollutant.drop_duplicates(inplace=True)
        for col in ["Good", "Normal", "Bad", "Very bad"]:
            if col in df_pollutant.columns:
                df_pollutant[col] = pd.to_numeric(df_pollutant[col], errors="coerce")
    
    # Q6: Count of Good, Normal, Bad, and Very bad records for PM2.5 pollutant
    if not df_pollutant.empty and "Item name" in df_pollutant.columns:
        df_pm25 = df_pollutant[df_pollutant["Item name"].str.contains("PM2.5", case=False, na=False)]
        if not df_pm25.empty:
            good_count = int(df_pm25["Good"].sum())
            normal_count = int(df_pm25["Normal"].sum())
            bad_count = int(df_pm25["Bad"].sum())
            very_bad_count = int(df_pm25["Very bad"].sum())
            q6 = {
                "Good": good_count,
                "Normal": normal_count,
                "Bad": bad_count,
                "Very bad": very_bad_count
            }
        else:
            q6 = {"Good": 0, "Normal": 0, "Bad": 0, "Very bad": 0}
    else:
        q6 = {"Good": 0, "Normal": 0, "Bad": 0, "Very bad": 0}
    
    result = {
        "target": {
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "Q4": q4,
            "Q5": q5,
            "Q6": q6
        }
    }
    return result

def main():
    result = compute_task1()
    if result is not None:
        output_path = os.path.join(BASE_DIR, "predictions/questions.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print("questions.json file has been created with the following content:")
        print(json.dumps(result, indent=2))
    else:
        print("Failed to compute task 1 results.")

if __name__ == "__main__":
    main()