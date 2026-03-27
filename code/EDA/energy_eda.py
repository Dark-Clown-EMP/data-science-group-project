import pandas as pd

# Correlate embedded solar generation with regional Solar_* features
INPUT_FILE = "final_model_data.csv"
OUTPUT_FILE = "embedded_solar_correlations.csv"

_df = pd.read_csv(INPUT_FILE, low_memory=False)

if "EMBEDDED_SOLAR_GENERATION" not in _df.columns:
    raise RuntimeError("EMBEDDED_SOLAR_GENERATION column not found in input file.")

_solar_cols = [c for c in _df.columns if c.startswith("Solar_")]
if not _solar_cols:
    raise RuntimeError("No Solar_* columns found in input file.")

_rows = []
for _col in _solar_cols:
    _pair = _df[["EMBEDDED_SOLAR_GENERATION", _col]].dropna()
    if _pair.empty:
        _corr = None
        _n = 0
    else:
        _corr = _pair["EMBEDDED_SOLAR_GENERATION"].corr(_pair[_col])
        _n = len(_pair)
    _rows.append({"solar_feature": _col, "corr_with_embedded_solar": _corr, "n_pairs": _n})

_out = pd.DataFrame(_rows).sort_values("corr_with_embedded_solar", ascending=False)
_out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved correlations to {OUTPUT_FILE}")
print("Top positive correlations:")
print(_out.head(5).to_string(index=False))
print("\nTop negative correlations:")
print(_out.tail(5).sort_values("corr_with_embedded_solar").to_string(index=False))
