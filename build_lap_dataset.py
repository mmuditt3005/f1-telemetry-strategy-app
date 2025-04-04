import fastf1
import pandas as pd

fastf1.Cache.enable_cache('f1_cache')

session = fastf1.get_session(2025, 'China', 'R')
session.load()

laps = session.laps
clean_laps = laps[laps['LapTime'].notnull()]

data = clean_laps[[
    'Driver', 'Team', 'Compound', 'LapNumber',
    'Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime'
]].copy()

def to_seconds(td):
    return td.total_seconds() if pd.notnull(td) else None

for col in ['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime']:
    data[col] = data[col].apply(to_seconds)

data.to_csv('china_2025_laps.csv', index=False)
print("✅ Dataset saved as 'china_2025_laps.csv'")
