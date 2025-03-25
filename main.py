from data.collector import fetch_game_data
from data.processor import process_data, add_features, encode_features, prepare_features
from models.trainer import train_model
from models.predictor import predict_upcoming_games
from config import DATA_PATH
import pandas as pd

def main():
    # Data collection
    print("Starting NBA Prediction Pipeline")
    raw_data = fetch_game_data()
    
    # Data processing
    processed_data = process_data(raw_data)
    processed_data = add_features(processed_data)
    processed_data = encode_features(processed_data)
    
    # Prepare features and target
    X, y = prepare_features(processed_data)
    
    # Train model
    model = train_model(X, y, processed_data['GAME_DATE'])
    
    # Save processed data
    processed_data.to_csv(DATA_PATH, index=False)
    print(f"Data saved to {DATA_PATH}")
    
    # Predict upcoming games
    predictions = predict_upcoming_games()
    if predictions is not None:
        print("\nUpcoming Game Predictions:")
        print(predictions[['GAME_DATE', 'MATCHUP', 'WIN_PROBABILITY', 'PREDICTED_WINNER']].to_string(index=False))

if __name__ == "__main__":
    main()