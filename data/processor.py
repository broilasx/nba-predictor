import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from ..config import TEAM_ENCODER_PATH, MATCHUP_ENCODER_PATH

def process_data(df):
    """Process and clean raw NBA game data"""
    # Select relevant columns
    df = df[['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION', 'MATCHUP', 
             'WL', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'PLUS_MINUS']]
    
    # Convert win/loss to binary
    df['WL'] = df['WL'].map({'W': 1, 'L': 0})
    
    # Convert date and sort
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df.sort_values('GAME_DATE', inplace=True)
    
    return df

def add_features(df):
    """Add rolling features to the dataset"""
    features = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'PLUS_MINUS']
    for feature in features:
        df[f'LAST5_{feature}'] = df.groupby('TEAM_ABBREVIATION')[feature].transform(
            lambda x: x.rolling(5, 1).mean().shift(1))
    return df.dropna()

def encode_features(df):
    """Encode categorical features"""
    le_team = LabelEncoder()
    le_matchup = LabelEncoder()
    
    df['TEAM_ABBREVIATION'] = le_team.fit_transform(df['TEAM_ABBREVIATION'])
    df['MATCHUP'] = le_matchup.fit_transform(df['MATCHUP'])
    
    # Save encoders
    joblib.dump(le_team, TEAM_ENCODER_PATH)
    joblib.dump(le_matchup, MATCHUP_ENCODER_PATH)
    
    return df

def prepare_features(df):
    """Prepare final feature set"""
    features = ['TEAM_ABBREVIATION', 'MATCHUP'] + [col for col in df.columns if col.startswith('LAST5_')]
    return df[features], df['WL']