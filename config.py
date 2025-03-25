# Configuration constants
SEASONS = [str(year) for year in range(2017, 2024)]
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PATH = 'models/nba_game_predictor.joblib'
DATA_PATH = 'data/nba_game_data.csv'
TEAM_ENCODER_PATH = 'models/team_encoder.joblib'
MATCHUP_ENCODER_PATH = 'models/matchup_encoder.joblib'

# Model parameters
PARAM_GRID = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}