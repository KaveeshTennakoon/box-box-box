import json
import sys
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path
from collections import defaultdict

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

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        c_stints = [s['len'] for s in stints if s['tire'] == compound]
        laps = sum(c_stints)
        wear = sum([L * (L + 1) / 2 for L in c_stints])
        wear_sq = sum([L * (L + 1) * (2 * L + 1) / 6 for L in c_stints])

        feats[f'{compound.lower()}_laps'] = laps
        feats[f'{compound.lower()}_wear'] = wear
        feats[f'{compound.lower()}_wear_sq'] = wear_sq

    return feats

def get_strategy_hash(strat):
    pit_stops = strat.get('pit_stops', [])
    stops_str = "_".join([f"lap{p['lap']}_to_{p['to_tire']}" for p in pit_stops])
    return f"{strat['starting_tire']}_{stops_str}"

def main():
    # Read from stdin
    input_data = sys.stdin.read()
    if not input_data: return
    test_case = json.loads(input_data)

    race_id = test_case['race_id']
    race_config = test_case['race_config']
    strategies = test_case['strategies']

    # Load artifacts
    model_path = Path("solution/model_xgb_final.pkl")
    meta_path = Path("solution/metadata_xgb_final.pkl")
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    model = joblib.load(model_path)
    feature_cols = joblib.load(meta_path)

    driver_list = []
    strategy_hashes = []
    sorted_keys = sorted(strategies.keys(), key=lambda x: int(x.replace('pos', '')))

    for pos_key in sorted_keys:
        strat = strategies[pos_key]
        feats = extract_ultimate_features(race_config, strat, pos_key)
        feats['driver_id'] = strat['driver_id']
        driver_list.append(feats)
        strategy_hashes.append(get_strategy_hash(strat))

    df_test = pd.DataFrame(driver_list)
    X_test = df_test[feature_cols]

    # Predict Physics Rank Score
    predicted_ranks = model.predict(X_test)

    # Strategy Clone Filter (Eliminates ML Floating Point Noise for identical strategies)
    hash_to_indices = defaultdict(list)
    for i, h in enumerate(strategy_hashes):
        hash_to_indices[h].append(i)

    for h, indices in hash_to_indices.items():
        if len(indices) > 1:
            scores = [predicted_ranks[i] for i in indices]
            scores.sort() # Lowest score is best rank
            for idx, score in zip(indices, scores):
                predicted_ranks[idx] = score

    results = []
    for i, d in enumerate(driver_list):
        results.append({
            'driver_id': d['driver_id'],
            'score': predicted_ranks[i],
            'start_pos': d['starting_pos']
        })

    # Sort mathematically
    results.sort(key=lambda x: (x['score'], x['start_pos']))
    finishing_positions = [x['driver_id'] for x in results]

    # Output pure JSON to stdout
    print(json.dumps({"race_id": race_id, "finishing_positions": finishing_positions}))

if __name__ == "__main__":
    main()