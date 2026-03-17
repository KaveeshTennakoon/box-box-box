import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

def engineer_features(race_config, strategy, pos_key):
    total_laps = race_config['total_laps']
    pit_stops = strategy.get('pit_stops', [])
    
    stints = []
    current_lap = 0
    current_tire = strategy['starting_tire']
    for stop in pit_stops:
        stints.append({'len': stop['lap'] - current_lap, 'tire': current_tire})
        current_lap = stop['lap']
        current_tire = stop['to_tire']
    stints.append({'len': total_laps - current_lap, 'tire': current_tire})
    
    soft_laps = sum(s['len'] for s in stints if s['tire'] == 'SOFT')
    medium_laps = sum(s['len'] for s in stints if s['tire'] == 'MEDIUM')
    hard_laps = sum(s['len'] for s in stints if s['tire'] == 'HARD')

    soft_wear = sum((s['len'] * (s['len'] + 1)) / 2 for s in stints if s['tire'] == 'SOFT')
    medium_wear = sum((s['len'] * (s['len'] + 1)) / 2 for s in stints if s['tire'] == 'MEDIUM')
    hard_wear = sum((s['len'] * (s['len'] + 1)) / 2 for s in stints if s['tire'] == 'HARD')

    track_temp = race_config.get('track_temp', 30.0)

    return {
        'starting_pos': int(pos_key.replace('pos', '')),
        'soft_laps': soft_laps,
        'medium_laps': medium_laps,
        'hard_laps': hard_laps,
        'soft_wear_temp': soft_wear * track_temp,
        'medium_wear_temp': medium_wear * track_temp,
        'hard_wear_temp': hard_wear * track_temp,
        'num_stops': len(pit_stops),
        'pit_time_loss': len(pit_stops) * race_config.get('pit_lane_time', 20.0)
    }

print("Loading historical races...")
data_dir = Path("data/historical_races")
all_races = []
for file in sorted(data_dir.glob("races_*.json")):
    with open(file) as f:
        all_races.extend(json.load(f))

data = []
for race in all_races:
    ranks = {driver: idx+1 for idx, driver in enumerate(race['finishing_positions'])}
    for pos_key, strat in race['strategies'].items():
        feats = engineer_features(race['race_config'], strat, pos_key)
        feats['target_rank'] = ranks[strat['driver_id']]
        data.append(feats)

df = pd.DataFrame(data)
print(f"Training dataset: {len(df)} rows")

X = df.drop('target_rank', axis=1)
y = df['target_rank']

imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("Training Smoothed Random Forest Regressor...")
# min_samples_leaf=3 smooths out random step-function noise
model = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=42, n_jobs=-1)
model.fit(X, y)

print(f"Training score (R^2): {model.score(X, y):.4f}")

solution_dir = Path("solution")
solution_dir.mkdir(exist_ok=True)
joblib.dump(model, solution_dir / "model_rf.pkl")
joblib.dump({
    'imputer': imputer,
    'feature_names': list(X.columns)
}, solution_dir / "metadata_rf.pkl")

print("✅ Perfected model saved!")