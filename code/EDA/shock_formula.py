import pandas as pd
import numpy as np

oilPriceDf = pd.read_csv('./data/oil_2015_2025.csv')
oilPriceDf.dropna()

eventDf = pd.read_csv('./data/OPEC events sentiment.csv')

prodWeightDf = pd.read_csv('./data/productionweights.csv')
prodWeightDf = prodWeightDf.drop(columns=['Country'])
prodWeightDf['Year'] = prodWeightDf['Year'] + 1

eventDf['Year'] = eventDf['Date'].astype('str').str[:4].astype('int')

weightMergedDf = pd.merge(prodWeightDf,
                   eventDf,
                   left_on=['Country Code', 'Year'],
                   right_on=['Country', 'Year'],
                   how='right')

#to be replaced by press freedom index dataset later
pressFreedomDict = {
    'AG': 65.8,  # Algeria
    'CF': 54.9,  # Congo
    'EK': 77.5,  # Equatorial Guinea
    'GB': 47.7,  # Gabon
    'IR': 87.6,  # Iran (Very high censorship)
    'IZ': 71.1,  # Iraq
    'KU': 61.4,  # Kuwait
    'LY': 74.9,  # Libya
    'NI': 41.3,  # Nigeria
    'SA': 84.7,  # Saudi Arabia (Very high censorship)
    'TC': 67.2,  # UAE
    'VE': 81.8,  # Venezuela
    'RS': 78.1,  # Russia
    'AO': 59.5   # Angola
}
pressFreedomDf = pd.DataFrame(list(pressFreedomDict.items()), columns=['Country', 'Press_Freedom_Index'])
# finalDf = pd.merge(weightMergedDf, pressFreedomDf,
#                    left_on=['Country'],
#                    right_on=['Country'],
#                    how='left')
finalDf = weightMergedDf.copy()
finalDf['ShockIndex'] = finalDf['Avg_Severity_Score'] * np.log1p(finalDf['Conflict_News_Volume'])\
    * finalDf['Production Weight']
print(finalDf['ShockIndex'])
# 1. Aggregate: Sum up all country shocks for each specific day
# This gives you ONE row per Date
dailyShock = finalDf.groupby('Date')['ShockIndex'].sum().reset_index()
dailyShock.to_csv('./data/shocks.csv', index= False)

# 2. Standardize Dates (as discussed)
dailyShock['Date'] = pd.to_datetime(dailyShock['Date'], format='%Y%m%d')
oilPriceDf['Date'] = pd.to_datetime(oilPriceDf['Date'])

# 3. Merge with Oil Prices
correlationDf = pd.merge(dailyShock, oilPriceDf, on='Date', how='inner')
# correlationDf['Price'] = correlationDf['Price'].ffill()
lagDf = correlationDf.copy()

lags = [1, 2, 3, 4]

# 4. The Moment of Truth
# Replace 'Price' with your actual column name (e.g., 'Value', 'DCOILWTICO')
print("\n--- GLOBAL CORRELATION lag 0 ---")
print(correlationDf[['ShockIndex', 'Price']].corr())


for lag in lags:
    colName = 'ShockIndexLag' + str(lag)
    lagDf['colName'] = lagDf['ShockIndex'].shift(lag)
    corr = lagDf['colName'].corr(lagDf['Price'])
    print(f"Lag {lag} Days: {corr}")

#considering weekend and weekday differently
correlationDfCase2 = pd.merge(dailyShock, oilPriceDf, on = 'Date', how='left')
correlationDfCase2['NextValidDate'] = correlationDfCase2['Date'].where(correlationDfCase2['Price'].notna()).bfill()
correlationDfCase2 = correlationDfCase2.dropna(subset=['NextValidDate'])

aggregatedDf = correlationDfCase2.groupby('NextValidDate').agg({
    'ShockIndex': 'sum',      # <--- This adds the missing shocks to the next day
    'Price': 'first',         # <--- Takes the actual Monday price
    'Date': 'count'           # <--- Counts how many days are in this bucket
}).rename(columns={'Date': 'DaysAccumulated'}).reset_index()

aggregatedDf.rename(columns={'NextValidDate': 'Date'}, inplace=True)

aggregatedDf['Day_OfWeek'] = aggregatedDf['Date'].dt.dayofweek

conditions = [
    # Case 1: Normal Weekday (Accumulated = 1 day. e.g., Tuesday News -> Tuesday Price)
    (aggregatedDf['DaysAccumulated'] == 1),
    
    # Case 3: Weekend Gap (Accumulated > 1 day AND it is Monday)
    (aggregatedDf['DaysAccumulated'] > 1) & (aggregatedDf['Day_OfWeek'] == 0),
    
    # Case 2: Holiday Gap (Accumulated > 1 day AND it is NOT Monday)
    (aggregatedDf['DaysAccumulated'] > 1) & (aggregatedDf['Day_OfWeek'] != 0)
]

choices = ['Case 1 (Normal)', 'Case 3 (Weekend)', 'Case 2 (Holiday)']
aggregatedDf['Scenario'] = np.select(conditions, choices, default='Unknown')

# 5. CHECK THE CORRELATION FOR EACH CASE
print("\n\n--- ANALYSIS RESULTS ---")
results = []

for scenario in ['Case 1 (Normal)', 'Case 3 (Weekend)', 'Case 2 (Holiday)']:
    subset = aggregatedDf[aggregatedDf['Scenario'] == scenario]
    
    if len(subset) > 10:
        corr = subset['ShockIndex'].corr(subset['Price'], method="spearman")
        print(f"{scenario}: Correlation = {corr} (Count: {len(subset)})")
        results.append((scenario, corr))
    else:
        print(f"{scenario}: Not enough data (Count: {len(subset)})")