from get_embedded_wind_solar import download_archives, OUTPUT_FILE


def main() -> None:
    df = download_archives()
    if "DATETIME" in df.columns:
        mask = (df["DATETIME"] >= "2020-01-01") & (df["DATETIME"] <= "2025-12-31")
        df = df.loc[mask]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
