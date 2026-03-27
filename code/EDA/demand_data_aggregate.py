import pandas as pd

year_list = ['2020', '2021', '2022', '2023', '2024', '2025']

def get_year_list():
    return year_list

csv_loc_init = './data/demanddata_'

for year in year_list:
    file_loc = csv_loc_init + year + '.csv'

    # 1. Load data
    df = pd.read_csv(file_loc)

    # 2. Create the Datetime column
    # We take the Date and add 30 minutes for every Period past the first one.
    df['datetime'] = pd.to_datetime(df['SETTLEMENT_DATE']) + \
                    pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 30, unit='min')

    # 3. Set Index
    df = df.set_index('datetime')

    # 4. Aggregate to Hourly (Mean)
    # This handles the 00:00 and 00:30 rows by averaging them into the 00:00 hour.
    df_hourly = df.select_dtypes(include='number').resample('1h').mean()

    # 5. Clean up
    df_hourly = df_hourly.drop(columns=['SETTLEMENT_PERIOD'])

    # 6. Save
    save_file_loc = './data/uk_grid_hourly_cleaned_' + year + '.csv'
    df_hourly.to_csv(save_file_loc)
    print("Done. Saved to " + save_file_loc)