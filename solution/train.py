import json
import glob
import pandas as pd
import xgboost as xgb
from pathlib import Path
from tqdm import tqdm
import joblib

def extract_ultimate_features(race_config, strategy, pos_key):
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

    feats = {
        'starting_pos': int(pos_key.replace('pos', '')),
        'num_stops': len(pit_stops),
        'track_temp': race_config.get('track_temp', 30.0),
        'base_lap_time': race_config.get('base_lap_time', 90.0),
        'pit_time_loss': len(pit_stops) * race_config.get('pit_lane_time', 20.0)
    }

    # Extract Polynomial Wear (Linear + Quadratic) to mathematically map the Tire Cliff
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        c_stints = [s['len'] for s in stints if s['tire'] == compound]
        laps = sum(c_stints)
        wear = sum([L * (L + 1) / 2 for L in c_stints])
        wear_sq = sum([L * (L + 1) * (2 * L + 1) / 6 for L in c_stints])

        feats[f'{compound.lower()}_laps'] = laps
        feats[f'{compound.lower()}_wear'] = wear
        feats[f'{compound.lower()}_wear_sq'] = wear_sq

    return feats

def main():
    print(" Loading Historical Race Data...")
    files = sorted(glob.glob("data/historical_races/races_*.json"))
    all_races = []
    for file in files:
        with open(file, 'r') as f:
            all_races.extend(json.load(f))

    data = []
    print("⚙️  Extracting Polynomial Physics Features...")
    for race in tqdm(all_races):
        ranks = {driver: idx + 1 for idx, driver in enumerate(race['finishing_positions'])}
        for pos_key, strat in race['strategies'].items():
            feats = extract_ultimate_features(race['race_config'], strat, pos_key)
            feats['target_rank'] = ranks[strat['driver_id']]
            data.append(feats)

    df = pd.DataFrame(data)

    feature_cols = [c for c in df.columns if c != 'target_rank']
    X = df[feature_cols]
    y = df['target_rank']

    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    print(f"Training Complete! Model R^2: {model.score(X, y):.5f}")

    # Ensure solution directory exists
    Path("solution").mkdir(exist_ok=True)
    
    joblib.dump(model, "solution/model_xgb_final.pkl")
    joblib.dump(feature_cols, "solution/metadata_xgb_final.pkl")
    print("Ready for Inference!")

if __name__ == "__main__":
    main()