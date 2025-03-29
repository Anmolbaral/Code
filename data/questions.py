import os
import pandas as pd
import numpy as np
import json

# Define base directory relative to this script's location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MEASUREMENT_FILE = os.path.join(BASE_DIR, "data/raw/measurement_data.csv")
INSTRUMENT_FILE = os.path.join(BASE_DIR, "data/raw/instrument_data.csv")
POLLUTANT_FILE = os.path.join(BASE_DIR, "data/raw/pollutant_data.csv")

def get_season(month):
    if month in [12, 1, 2]:
        return "1"  # Winter
    elif month in [3, 4, 5]:
        return "2"  # Spring
    elif month in [6, 7, 8]:
        return "3"  # Summer
    else:
        return "4"  # Autumn

def preprocess_measurement_data(df):
    # Drop duplicates and rows missing Measurement date
    df = df.drop_duplicates().dropna(subset=["Measurement date"])
    # Convert pollutant columns to numeric
    pollutant_cols = ["SO2", "NO2", "O3", "CO", "PM10", "PM2.5"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Convert 'Station code' to numeric and drop rows where conversion fails
    df["Station code"] = pd.to_numeric(df["Station code"], errors="coerce")
    df = df.dropna(subset=["Station code"])
    # Ensure Measurement date is a datetime
    df["Measurement date"] = pd.to_datetime(df["Measurement date"], errors="coerce")
    return df

def preprocess_instrument_data(df):
    # Drop duplicates and rows missing critical columns
    df = df.drop_duplicates().dropna(subset=["Measurement date", "Station code", "Instrument status"])
    # Trim whitespace in Instrument status if necessary
    if df["Instrument status"].dtype == object:
        df["Instrument status"] = df["Instrument status"].str.strip()
    # Convert columns to numeric
    df["Station code"] = pd.to_numeric(df["Station code"], errors="coerce")
    df["Instrument status"] = pd.to_numeric(df["Instrument status"], errors="coerce")
    return df.dropna(subset=["Station code", "Instrument status"])

def load_pollutant_mapping():
    try:
        df = pd.read_csv(POLLUTANT_FILE)
        # Assume columns 'Item name' and 'Item code' exist; trim whitespace in column names
        df.columns = [c.strip() for c in df.columns]
        return df.set_index('Item name')['Item code'].to_dict()
    except Exception as e:
        print("Error loading pollutant data:", e)
        return {}

def compute_task1():
    # Load measurement data
    try:
        df_measure = pd.read_csv(MEASUREMENT_FILE, parse_dates=["Measurement date"])
    except Exception as e:
        print("Error loading measurement data:", e)
        return {}
    df_measure = preprocess_measurement_data(df_measure)
    
    # Load instrument data
    try:
        df_instrument = pd.read_csv(INSTRUMENT_FILE, parse_dates=["Measurement date"])
    except Exception as e:
        print("Error loading instrument data:", e)
        return {}
    df_instrument = preprocess_instrument_data(df_instrument)
    
    pollutant_map = load_pollutant_mapping()
    
    # Q1: Average daily SO2 concentration across all stations.
    q1 = 0.0
    so2_code = pollutant_map.get('SO2')
    if so2_code is not None:
        # Filter instrument data for SO2 when status is 0
        instr_so2 = df_instrument[(df_instrument['Item code'] == so2_code) & (df_instrument['Instrument status'] == 0)]
        # Merge on Station code and Measurement date
        merged = pd.merge(df_measure, instr_so2[['Station code', 'Measurement date']], 
                          on=['Station code', 'Measurement date'], how='inner')
        merged = merged.dropna(subset=['SO2'])
        if not merged.empty:
            # Group by Station code and date
            daily_avg = merged.groupby(['Station code', merged['Measurement date'].dt.date])['SO2'].mean()
            q1 = round(daily_avg.mean(), 5)
    
    # Q2: Average CO per season at station 209.
    q2 = {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}
    co_code = pollutant_map.get('CO')
    if co_code is not None:
        instr_co = df_instrument[(df_instrument['Item code'] == co_code) & (df_instrument['Instrument status'] == 0)]
        merged = pd.merge(df_measure, instr_co[['Station code', 'Measurement date']], 
                          on=['Station code', 'Measurement date'], how='inner')
        merged = merged[merged['Station code'] == 209].dropna(subset=['CO'])
        if not merged.empty:
            merged['month'] = merged['Measurement date'].dt.month
            merged['season'] = merged['month'].apply(get_season)
            q2 = merged.groupby('season')['CO'].mean().round(5).to_dict()
            # Convert numpy numbers to Python floats
            q2 = {k: float(v) for k, v in q2.items()}
    
    # Q3: Hour with highest variability (standard deviation) for O3.
    q3 = 0
    o3_code = pollutant_map.get('O3')
    if o3_code is not None:
        instr_o3 = df_instrument[(df_instrument['Item code'] == o3_code) & (df_instrument['Instrument status'] == 0)]
        merged = pd.merge(df_measure, instr_o3[['Station code', 'Measurement date']], 
                          on=['Station code', 'Measurement date'], how='inner')
        merged = merged.dropna(subset=['O3'])
        if not merged.empty:
            merged['hour'] = merged['Measurement date'].dt.hour
            q3 = int(merged.groupby('hour')['O3'].std().idxmax())
    
    # Q4: Station with most 'Abnormal data' (Instrument status 9)
    abnormal = df_instrument[df_instrument['Instrument status'] == 9]
    q4 = int(abnormal['Station code'].value_counts().idxmax()) if not abnormal.empty else None
    
    # Q5: Station with most 'not normal' measurements (Instrument status != 0)
    not_normal = df_instrument[df_instrument['Instrument status'] != 0]
    q5 = int(not_normal['Station code'].value_counts().idxmax()) if not not_normal.empty else None
    
    # Q6: Count PM2.5 records by quality category.
    q6 = {"Good": 0, "Normal": 0, "Bad": 0, "Very bad": 0}
    pm25_code = pollutant_map.get('PM2.5')
    if pm25_code is not None:
        try:
            pm25_info = pd.read_csv(POLLUTANT_FILE)
            pm25_info.columns = [c.strip() for c in pm25_info.columns]
        except Exception as e:
            print("Error re-loading pollutant data for Q6:", e)
            pm25_info = pd.DataFrame()
        if not pm25_info.empty and 'Item name' in pm25_info.columns:
            pm25_row = pm25_info[pm25_info['Item name'] == 'PM2.5']
            if not pm25_row.empty:
                pm25_row = pm25_row.iloc[0]
                good = pm25_row['Good']
                normal = pm25_row['Normal']
                bad = pm25_row['Bad']
                # Filter instrument data for PM2.5 when status is 0
                instr_pm25 = df_instrument[(df_instrument['Item code'] == pm25_code) & 
                                           (df_instrument['Instrument status'] == 0)]
                merged = pd.merge(df_measure, instr_pm25[['Station code', 'Measurement date']], 
                                  on=['Station code', 'Measurement date'], how='inner')
                merged = merged.dropna(subset=['PM2.5'])
                pm25 = merged['PM2.5']
                q6 = {
                    "Good": int(((pm25 <= good)).sum()),
                    "Normal": int(((pm25 > good) & (pm25 <= normal)).sum()),
                    "Bad": int(((pm25 > normal) & (pm25 <= bad)).sum()),
                    "Very bad": int(((pm25 > bad)).sum())
                }
    
    # Ensure that all numpy types are converted to native Python types
    result = {
        "target": {
            "Q1": float(q1),
            "Q2": q2,
            "Q3": q3,
            "Q4": int(q4) if q4 is not None else None,
            "Q5": int(q5) if q5 is not None else None,
            "Q6": {k: int(v) for k, v in q6.items()}
        }
    }
    return result

def main():
    result = compute_task1()
    output_path = os.path.join(BASE_DIR, "predictions/questions.json")
    try:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print("questions.json created at", output_path)
    except Exception as e:
        print("Error writing questions.json:", e)

if __name__ == "__main__":
    main()
