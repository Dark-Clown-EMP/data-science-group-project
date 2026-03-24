"""Feature engineering utilities for energy demand forecasting."""

import pandas as pd


WEATHER_COLS = [
    "Temp_Scot_Highlands", "Wind10m_Scot_Highlands",
    "Temp_Scot_Aberdeenshire", "Wind10m_Scot_Aberdeenshire",
    "Temp_Scot_Glasgow_West", "Temp_Scot_Edinburgh_East",
    "Temp_Scot_Borders", "Wind10m_Scot_Borders",
    "Temp_Wales_North_Gwynedd", "Wind10m_Wales_North_Gwynedd",
    "Temp_Wales_South_Cardiff",
    "Temp_Eng_North_Tyne", "Temp_Eng_North_Cumbria", "Wind10m_Eng_North_Cumbria",
    "Temp_Eng_Yorkshire", "Wind10m_Eng_Yorkshire",
    "Temp_Eng_Manchester", "Temp_Eng_West_Midlands",
    "Temp_Eng_East_Midlands", "Solar_Eng_East_Midlands",
    "Temp_Eng_East_Norfolk", "Wind10m_Eng_East_Norfolk", "Solar_Eng_East_Norfolk",
    "Temp_Eng_East_Suffolk", "Wind10m_Eng_East_Suffolk", "Solar_Eng_East_Suffolk",
    "Temp_Eng_London", "Solar_Eng_London",
    "Temp_Eng_South_Kent", "Solar_Eng_South_Kent",
    "Temp_Eng_South_Hampshire", "Solar_Eng_South_Hampshire",
    "Temp_Eng_South_Cornwall", "Solar_Eng_South_Cornwall",
    "Temp_Eng_South_Bristol", "Solar_Eng_South_Bristol",
]

DEMAND_COLS = [
    "TSD", "ENGLAND_WALES_DEMAND",
    "EMBEDDED_WIND_GENERATION", "EMBEDDED_WIND_CAPACITY",
    "EMBEDDED_SOLAR_GENERATION", "EMBEDDED_SOLAR_CAPACITY",
    "NON_BM_STOR", "PUMP_STORAGE_PUMPING",
    "NET_IMPORTS", "SCOTTISH_TRANSFER",
]


def _add_time_features(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def prepare_model_frame(df, include_weather=True, target_col="ND"):
    df = _add_time_features(df)

    time_features = ["hour", "day_of_week", "month", "day_of_year", "is_weekend"]

    available_demand = [c for c in DEMAND_COLS if c in df.columns]
    feature_cols = time_features + available_demand

    if include_weather:
        available_weather = [c for c in WEATHER_COLS if c in df.columns]
        feature_cols += available_weather

    keep_cols = ["datetime", target_col] + feature_cols
    df_model = df[keep_cols].dropna().reset_index(drop=True)

    return df_model, feature_cols
