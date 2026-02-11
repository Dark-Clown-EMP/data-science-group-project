import pandas as pd

# Correlate embedded wind generation with regional Wind10m_* features
INPUT_FILE = "final_model_data.csv"
OUTPUT_FILE = "embedded_wind_correlations.csv"

_df = pd.read_csv(INPUT_FILE, low_memory=False)

if "EMBEDDED_WIND_GENERATION" not in _df.columns:
    raise RuntimeError("EMBEDDED_WIND_GENERATION column not found in input file.")

_wind_cols = [c for c in _df.columns if c.startswith("Wind10m_")]
if not _wind_cols:
    raise RuntimeError("No Wind10m_* columns found in input file.")

_rows = []
for _col in _wind_cols:
    _pair = _df[["EMBEDDED_WIND_GENERATION", _col]].dropna()
    if _pair.empty:
        _corr = None
        _n = 0
    else:
        _corr = _pair["EMBEDDED_WIND_GENERATION"].corr(_pair[_col])
        _n = len(_pair)
    _rows.append({"wind_feature": _col, "corr_with_embedded_wind": _corr, "n_pairs": _n})

_out = pd.DataFrame(_rows).sort_values("corr_with_embedded_wind", ascending=False)
_out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved correlations to {OUTPUT_FILE}")
print("Top positive correlations:")
print(_out.head(5).to_string(index=False))
print("\nTop negative correlations:")
print(_out.tail(5).sort_values("corr_with_embedded_wind").to_string(index=False))
