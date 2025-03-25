from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams
import pandas as pd
import joblib
import os
from config import MODEL_PATH, TEAM_ENCODER_PATH, MATCHUP_ENCODER_PATH  

def predict_upcoming_games():
    """Predict outcomes for upcoming NBA games"""
    print("\nFetching upcoming games...")
    
    try:
        # 1. Load required models and encoders
        model = joblib.load(MODEL_PATH)
        le_team = joblib.load(TEAM_ENCODER_PATH)
        le_matchup = joblib.load(MATCHUP_ENCODER_PATH)

        # 2. Fetch scoreboard data
        scoreboard = ScoreboardV2()
        games_df = scoreboard.game_header.get_data_frame()
        
        if games_df.empty:
            print("No upcoming games found.")
            return None

        # 3. Process game data
        games = games_df.copy()
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE_EST']).dt.strftime('%Y-%m-%d')
        
        # Get team abbreviations
        nba_teams = teams.get_teams()
        team_id_to_abbr = {team['id']: team['abbreviation'] for team in nba_teams}
        
        games['HOME_TEAM_ABBR'] = games['HOME_TEAM_ID'].map(team_id_to_abbr)
        games['VISITOR_TEAM_ABBR'] = games['VISITOR_TEAM_ID'].map(team_id_to_abbr)
        games['MATCHUP'] = games['VISITOR_TEAM_ABBR'] + ' @ ' + games['HOME_TEAM_ABBR']

        # Create separate rows for home and away teams
        home = games.copy()
        home['TEAM_ABBREVIATION'] = home['HOME_TEAM_ABBR']
        home['IS_HOME'] = 1
        
        away = games.copy()
        away['TEAM_ABBREVIATION'] = away['VISITOR_TEAM_ABBR']
        away['IS_HOME'] = 0
        
        combined = pd.concat([home, away])

        # 4. Transform features
        combined['TEAM_ABBREVIATION'] = le_team.transform(combined['TEAM_ABBREVIATION'])
        combined['MATCHUP'] = le_matchup.transform(combined['MATCHUP'])

        # 5. Prepare feature set (must match training features)
        features = [
            'TEAM_ABBREVIATION', 'MATCHUP',
            'LAST5_PTS', 'LAST5_REB', 'LAST5_AST', 'LAST5_STL',
            'LAST5_BLK', 'LAST5_TOV', 'LAST5_FG_PCT', 'LAST5_FT_PCT',
            'LAST5_FG3_PCT', 'LAST5_PLUS_MINUS'
        ]
        
        # Add missing features with default values
        for col in features:
            if col not in combined.columns:
                combined[col] = 0.0

        # 6. Make predictions
        pred_proba = model.predict_proba(combined[features])[:, 1]
        combined['WIN_PROBABILITY'] = pred_proba
        combined['PREDICTED_WINNER'] = (pred_proba > 0.5).astype(int)
        combined['PREDICTION'] = combined['PREDICTED_WINNER'].map({1: 'Win', 0: 'Loss'})

        # 7. Prepare output
        combined['TEAM_ABBREVIATION'] = le_team.inverse_transform(combined['TEAM_ABBREVIATION'])
        combined['MATCHUP'] = le_matchup.inverse_transform(combined['MATCHUP'])
        
        output_cols = [
            'GAME_DATE', 'MATCHUP', 'TEAM_ABBREVIATION', 
            'IS_HOME', 'WIN_PROBABILITY', 'PREDICTION'
        ]
        
        results = combined[output_cols].rename(columns={
            'TEAM_ABBREVIATION': 'TEAM',
            'PREDICTION': 'PREDICTED_OUTCOME'
        }).sort_values(['GAME_DATE', 'MATCHUP', 'IS_HOME'])
        
        return results

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None