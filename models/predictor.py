from nba_api.stats.endpoints import Scoreboard
import pandas as pd
import joblib
from ..config import MODEL_PATH, TEAM_ENCODER_PATH, MATCHUP_ENCODER_PATH

def predict_upcoming_games():
    """Predict outcomes for upcoming NBA games"""
    print("\nFetching upcoming games...")
    try:
        # Load model and encoders
        model = joblib.load(MODEL_PATH)
        le_team = joblib.load(TEAM_ENCODER_PATH)
        le_matchup = joblib.load(MATCHUP_ENCODER_PATH)
        
        # Get upcoming games
        scoreboard = Scoreboard()
        games = scoreboard.get_data_frames()[0]
        
        if games.empty:
            print("No upcoming games found.")
            return None
        
        # Prepare features (simplified - in practice you'd need recent team stats)
        games['TEAM_ABBREVIATION'] = le_team.transform(games['TEAM_ABBREVIATION'])
        games['MATCHUP'] = le_matchup.transform(games['MATCHUP'])
        
        # For demo purposes - in reality you'd need actual recent stats
        features = ['TEAM_ABBREVIATION', 'MATCHUP'] + [f'LAST5_{f}' for f in 
                    ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'PLUS_MINUS']]
        
        # Add placeholder features (in real app, fetch actual recent stats)
        for feature in features[2:]:  # Skip first two which are categorical
            games[feature] = 0  # Placeholder - should be actual rolling averages
        
        # Make predictions
        predictions = model.predict_proba(games[features])
        games['WIN_PROBABILITY'] = predictions[:, 1]
        games['PREDICTED_WINNER'] = (games['WIN_PROBABILITY'] > 0.5).astype(int)
        
        # Convert back to team abbreviations
        games['TEAM_ABBREVIATION'] = le_team.inverse_transform(games['TEAM_ABBREVIATION'])
        games['MATCHUP'] = le_matchup.inverse_transform(games['MATCHUP'])
        
        return games
    
    except Exception as e:
        print(f"Error predicting upcoming games: {e}")
        return None