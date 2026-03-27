import argparse
import pandas as pd


def split_gsp_names(s: str):
    if pd.isna(s):
        return []
    return [p.strip() for p in str(s).split("|") if p.strip()]


def main():
    parser = argparse.ArgumentParser(description="Join PV-Live solar data to NESO GSP info with defensible mapping.")
    parser.add_argument("--solar", default="regional_solar_2020_2025.csv", help="PV-Live solar CSV")
    parser.add_argument("--gsp", default="gsp_info.csv", help="NESO GSP info CSV")
    parser.add_argument("--out", default="regional_solar_2020_2025_labeled.csv", help="Output CSV")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate back to original gsp_name after mapping")
    args = parser.parse_args()

    solar = pd.read_csv(args.solar)
    gsp = pd.read_csv(args.gsp)

    # Normalize columns
    solar.columns = [c.strip() for c in solar.columns]
    gsp.columns = [c.strip() for c in gsp.columns]

    if "gsp_name" not in solar.columns:
        raise RuntimeError("solar file missing gsp_name column")
    if "GSP ID" not in gsp.columns:
        raise RuntimeError("gsp_info file missing 'GSP ID' column")

    # Build mapping table from gsp_info
    gsp_map = gsp[["GSP ID", "Name", "GSP Group", "Latitude", "Longitude", "Comments"]].copy()
    gsp_map["GSP ID"] = gsp_map["GSP ID"].astype(str).str.strip()

    # Expand pipe-separated gsp_name into rows
    solar_exp = solar.copy()
    solar_exp["_gsp_parts"] = solar_exp["gsp_name"].apply(split_gsp_names)
    solar_exp = solar_exp.explode("_gsp_parts")
    solar_exp["_gsp_parts"] = solar_exp["_gsp_parts"].astype(str).str.strip()

    # Join on individual GSP ID
    merged = solar_exp.merge(gsp_map, left_on="_gsp_parts", right_on="GSP ID", how="left")

    # Quality report
    total = len(merged)
    matched = merged["GSP ID"].notna().sum()
    print(f"Rows after expansion: {total}")
    print(f"Matched rows: {matched} ({matched/total:.2%})")

    if args.aggregate:
        # Aggregate back to original gsp_name while keeping a readable label
        group_cols = [c for c in solar.columns if c not in ["gsp_id", "gsp_name"]]
        value_cols = [c for c in solar.columns if c in ["generation_mw"]]
        # Default to summing generation_mw if present
        if "generation_mw" in merged.columns:
            agg = merged.groupby(group_cols + ["gsp_name"], dropna=False)["generation_mw"].sum().reset_index()
        else:
            agg = merged.groupby(group_cols + ["gsp_name"], dropna=False).size().reset_index(name="row_count")
        agg.to_csv(args.out, index=False)
        print(f"Saved aggregated file to {args.out}")
        return

    merged.to_csv(args.out, index=False)
    print(f"Saved expanded+joined file to {args.out}")


if __name__ == "__main__":
    main()
