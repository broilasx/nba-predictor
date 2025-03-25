from data.collector import fetch_game_data
from data.processor import process_data, add_features, encode_features, prepare_features
from models.trainer import train_model
from models.predictor import predict_upcoming_games
from config import DATA_PATH, MODEL_PATH
import pandas as pd
import os
import time

def main():
    print("Starting NBA Prediction Pipeline")
    
    # Data collection
    raw_data = fetch_game_data()
    
    # Data processing
    processed_data = process_data(raw_data)
    processed_data = add_features(processed_data)
    processed_data = encode_features(processed_data)
    
    # Prepare features and target
    X, y = prepare_features(processed_data)
    
    # Model training with completion check
    model_trained = False
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training new model...")
        try:
            start_time = time.time()
            model = train_model(X, y, processed_data['GAME_DATE'])
            training_time = time.time() - start_time
            print(f"Model training completed in {training_time:.2f} seconds")
            model_trained = True
        except Exception as e:
            print(f"Model training failed: {e}")
            return
    
    # Only attempt predictions if model was trained or exists
    if model_trained or os.path.exists(MODEL_PATH):
        print("\nAttempting predictions...")
        try:
            predictions = predict_upcoming_games()
            if predictions is not None:
                print("\nUpcoming Game Predictions:")
                print(predictions[['GAME_DATE', 'MATCHUP', 'WIN_PROBABILITY', 'PREDICTED_WINNER']].to_string(index=False))
        except Exception as e:
            print(f"\nPrediction error: {e}")
    else:
        print("\nSkipping predictions - no trained model available")
    
    # Save processed data
    processed_data.to_csv(DATA_PATH, index=False)
    print(f"\nData saved to {DATA_PATH}")

if __name__ == "__main__":
    main()