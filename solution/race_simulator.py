import json
import sys
import pandas as pd
import joblib
from pathlib import Path
from collections import defaultdict

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

# NEW: Create a unique fingerprint for a strategy
def get_strategy_hash(strat):
    pit_stops = strat.get('pit_stops', [])
    stops_str = "_".join([f"lap{p['lap']}_to_{p['to_tire']}" for p in pit_stops])
    return f"{strat['starting_tire']}_{stops_str}"

def main():
    input_data = sys.stdin.read()
    if not input_data: return
    test_case = json.loads(input_data)

    race_id = test_case['race_id']
    race_config = test_case['race_config']
    strategies = test_case['strategies']

    metadata = joblib.load(Path("solution/metadata_rf.pkl"))
    model = joblib.load(Path("solution/model_rf.pkl"))
    imputer = metadata['imputer']
    feature_names = metadata['feature_names']

    driver_list = []
    strategy_hashes = []
    sorted_keys = sorted(strategies.keys(), key=lambda x: int(x.replace('pos', '')))
    
    for pos_key in sorted_keys:
        strat = strategies[pos_key]
        feats = engineer_features(race_config, strat, pos_key)
        feats['driver_id'] = strat['driver_id']
        driver_list.append(feats)
        strategy_hashes.append(get_strategy_hash(strat))

    df = pd.DataFrame(driver_list)
    X = df[feature_names]
    X_imputed = pd.DataFrame(imputer.transform(X), columns=feature_names)

    predicted_ranks = model.predict(X_imputed)
    
    # NEW: Filter and enforce grid logic for Strategy Clones
    hash_to_indices = defaultdict(list)
    for i, h in enumerate(strategy_hashes):
        hash_to_indices[h].append(i)
        
    for h, indices in hash_to_indices.items():
        if len(indices) > 1:
            # Gather all ML scores for this specific cloned strategy
            scores = [predicted_ranks[i] for i in indices]
            # Sort scores ascending (lowest rank is best)
            scores.sort()
            # Because 'indices' is already sorted by starting grid position,
            # this loop perfectly pairs the best score with the highest grid position.
            for idx, score in zip(indices, scores):
                predicted_ranks[idx] = score

    results = []
    for i, d in enumerate(driver_list):
        results.append({
            'driver_id': d['driver_id'], 
            'score': predicted_ranks[i],
            'start_pos': d['starting_pos']
        })

    # Sort final results
    results.sort(key=lambda x: (x['score'], x['start_pos']))
    
    finishing = [x['driver_id'] for x in results]
    print(json.dumps({"race_id": race_id, "finishing_positions": finishing}))

if __name__ == "__main__":
    main()