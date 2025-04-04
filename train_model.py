import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

df = pd.read_csv('china_2025_laps.csv')
df.dropna(inplace=True)

print("🧾 Drivers:", df['Driver'].unique())
print("🧾 Teams:", df['Team'].unique())
print("🧾 Compounds:", df['Compound'].unique())

driver_enc = LabelEncoder()
team_enc = LabelEncoder()
compound_enc = LabelEncoder()

df['Driver_enc'] = driver_enc.fit_transform(df['Driver'])
df['Team_enc'] = team_enc.fit_transform(df['Team'])
df['Compound_enc'] = compound_enc.fit_transform(df['Compound'])

X = df[['Driver_enc', 'Team_enc', 'Compound_enc', 'LapNumber']]
y = df['LapTime']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (Random Forest this time!)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✅ Model trained!")
print(f"📊 MAE: {mae:.3f} sec")
print(f"📊 RMSE: {rmse:.3f} sec")

# Save model + encoders
joblib.dump(model, 'lap_model_nosectors.pkl')
joblib.dump(driver_enc, 'Driver_encoder_nosectors.pkl')
joblib.dump(team_enc, 'Team_encoder_nosectors.pkl')
joblib.dump(compound_enc, 'Compound_encoder_nosectors.pkl')

print("\n🔮 Predicting VER | Red Bull | HARD | lap 12")

test_input = pd.DataFrame([{
    'Driver_enc': driver_enc.transform(['VER'])[0],
    'Team_enc': team_enc.transform(['Red Bull Racing'])[0],
    'Compound_enc': compound_enc.transform(['HARD'])[0],
    'LapNumber': 12
}])

prediction = model.predict(test_input)[0]
print(f"🎯 Predicted Lap Time (no sectors): {prediction:.3f} sec")
