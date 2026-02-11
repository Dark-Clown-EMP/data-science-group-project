import pandas as pd

# Correlate TSD with regional temperature (Temp_*) features
INPUT_FILE = "final_model_data.csv"
OUTPUT_FILE = "tsd_temp_correlations.csv"

_df = pd.read_csv(INPUT_FILE, low_memory=False)

if "TSD" not in _df.columns:
    raise RuntimeError("TSD column not found in input file.")

_temp_cols = [c for c in _df.columns if c.startswith("Temp_")]
if not _temp_cols:
    raise RuntimeError("No Temp_* columns found in input file.")

_rows = []
for _col in _temp_cols:
    _pair = _df[["TSD", _col]].dropna()
    if _pair.empty:
        _corr = None
        _n = 0
    else:
        _corr = _pair["TSD"].corr(_pair[_col])
        _n = len(_pair)
    _rows.append({"temperature_feature": _col, "corr_with_TSD": _corr, "n_pairs": _n})

_out = pd.DataFrame(_rows).sort_values("corr_with_TSD", ascending=False)
_out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved correlations to {OUTPUT_FILE}")
print("Top positive correlations:")
print(_out.head(5).to_string(index=False))
print("\nTop negative correlations:")
print(_out.tail(5).sort_values("corr_with_TSD").to_string(index=False))
