import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from config import TEAM_ENCODER_PATH, MATCHUP_ENCODER_PATH

def process_data(raw_df):
    """Process and clean raw NBA game data"""
    # Create a copy of the input DataFrame to avoid warnings
    df = raw_df.copy()
    
    # Select relevant columns
    df = df[['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION', 'MATCHUP', 
             'WL', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'PLUS_MINUS']]
    
    # Convert win/loss to binary - with validation
    if not set(df['WL'].unique()).issubset({'W', 'L'}):
        raise ValueError("WL column contains unexpected values")
    
    df.loc[:, 'WL'] = df['WL'].map({'W': 1, 'L': 0}).astype(int)
    
    # Convert date and sort - using .loc and avoiding inplace
    df.loc[:, 'GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')  # Don't use inplace=True
    
    return df

def add_features(df):
    """Add rolling features and home/away indicator"""
    df = df.copy()
    
    # Add home/away indicator (assuming MATCHUP format is "XXX @ YYY")
    df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    # Add rolling stats
    features = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'PLUS_MINUS']
    for feature in features:
        df[f'LAST5_{feature}'] = df.groupby('TEAM_ABBREVIATION')[feature].transform(
            lambda x: x.rolling(5, 1).mean().shift(1))
    
    return df.dropna()

def encode_features(df):
    """Encode categorical features"""
    # Create a fresh copy to work with
    df = df.copy()
    
    le_team = LabelEncoder()
    le_matchup = LabelEncoder()
    
    df.loc[:, 'TEAM_ABBREVIATION'] = le_team.fit_transform(df['TEAM_ABBREVIATION'])
    df.loc[:, 'MATCHUP'] = le_matchup.fit_transform(df['MATCHUP'])
    
    # Save encoders
    joblib.dump(le_team, TEAM_ENCODER_PATH)
    joblib.dump(le_matchup, MATCHUP_ENCODER_PATH)
    
    return df

def prepare_features(df):
    """Prepare final feature set"""
    features = ['TEAM_ABBREVIATION', 'MATCHUP'] + [col for col in df.columns if col.startswith('LAST5_')]
    X = df[features]
    y = df['WL'].astype(int)  # Ensure integer type
    
    # Validate we have binary targets
    if set(y.unique()) != {0, 1}:
        raise ValueError("Target variable must contain only 0 and 1 values")
    
    return X, y