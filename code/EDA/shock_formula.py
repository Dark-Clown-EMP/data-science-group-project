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
finalDf = pd.merge(weightMergedDf, pressFreedomDf,
                   left_on=['Country'],
                   right_on=['Country'],
                   how='left')
finalDf['ShockIndex'] = finalDf['Avg_Severity_Score'] * np.log1p(finalDf['Conflict_News_Volume'])\
    * finalDf['Press_Freedom_Index'] * finalDf['Production Weight']
print(finalDf['ShockIndex'])
finalDf.to_csv('./data/shock.csv', index= False)
# 1. Aggregate: Sum up all country shocks for each specific day
# This gives you ONE row per Date
daily_shock = finalDf.groupby('Date')['ShockIndex'].sum().reset_index()

# 2. Standardize Dates (as discussed)
daily_shock['Date'] = pd.to_datetime(daily_shock['Date'], format='%Y%m%d')
oilPriceDf['Date'] = pd.to_datetime(oilPriceDf['Date'])

# 3. Merge with Oil Prices
correlation_df = pd.merge(daily_shock, oilPriceDf, on='Date', how='inner')

# 4. The Moment of Truth
# Replace 'Price' with your actual column name (e.g., 'Value', 'DCOILWTICO')
print("\n--- GLOBAL CORRELATION ---")
print(correlation_df[['ShockIndex', 'Price']].corr())