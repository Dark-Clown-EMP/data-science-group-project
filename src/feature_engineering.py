import numpy as np
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

LAG_HOURS = [24, 48, 72, 168]
ROLLING_WINDOWS = [24, 48, 168]


def _add_time_features(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_lag_features(df, target_col="ND", lags=LAG_HOURS):
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)
    return df


def _add_rolling_features(df, target_col="ND", windows=ROLLING_WINDOWS):
    df = df.copy()
    for w in windows:
        df[f"{target_col}_rmean_{w}h"] = df[target_col].shift(24).rolling(w).mean()
        df[f"{target_col}_rstd_{w}h"] = df[target_col].shift(24).rolling(w).std()
    return df


def prepare_model_frame(df, include_weather=True, target_col="ND",
                        lag_hours=None, rolling_windows=None):
    if lag_hours is None:
        lag_hours = LAG_HOURS
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    df = _add_time_features(df)
    df = _add_lag_features(df, target_col=target_col, lags=lag_hours)
    df = _add_rolling_features(df, target_col=target_col, windows=rolling_windows)

    time_features = [
        "hour", "day_of_week", "month", "day_of_year", "is_weekend",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
    ]
    lag_features = [f"{target_col}_lag_{lag}h" for lag in lag_hours]
    rolling_features = [
        f"{target_col}_{stat}_{w}h"
        for w in rolling_windows for stat in ["rmean", "rstd"]
    ]

    feature_cols = time_features + lag_features + rolling_features

    if include_weather:
        weather_lag = 24
        lagged_weather_cols = {}
        for col in WEATHER_COLS:
            if col in df.columns:
                lagged_weather_cols[f"{col}_lag_{weather_lag}h"] = df[col].shift(weather_lag)
        if lagged_weather_cols:
            df = pd.concat([df, pd.DataFrame(lagged_weather_cols)], axis=1)
        lagged_weather = list(lagged_weather_cols.keys())
        feature_cols += lagged_weather

    keep_cols = ["datetime", target_col] + feature_cols
    df_model = df[keep_cols].dropna().reset_index(drop=True)

    return df_model, feature_cols
