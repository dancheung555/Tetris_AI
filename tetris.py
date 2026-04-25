import numpy as np
import sys
import time
import pandas as pd
from tetramino_generator import generate_next_bag
from tetramino_generator import generate_n_bags
from lstm_predictor import TetrominoLSTMPredictor

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 1

def train_lstm_model(n_bags=100, epochs=30, batch_size=32, save_model=True):
    """Train the LSTM model for tetromino prediction"""
    print(f"Initializing LSTM predictor with sliding window of 6...")
    predictor = TetrominoLSTMPredictor(window_size=6)
    
    print(f"Training LSTM model on {n_bags} bags for {epochs} epochs...")
    predictor.train(n_bags=n_bags, epochs=epochs, batch_size=batch_size)
    
    if save_model:
        model_path = 'lstm_tetromino_model.pth'
        predictor.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    return predictor

def predict_next_bags(predictor, base_sequence, n_bags=20):
    """Predict the next n bags using the trained model"""
    flat_sequence = [piece for bag in base_sequence for piece in bag]
    last_6_pieces = flat_sequence[-6:]
    
    print(f"\nPredicting next {n_bags} bags...")
    predicted_pieces = []
    current_sequence = list(last_6_pieces)
    
    for i in range(n_bags * 7):  # n_bags * 7 pieces per bag
        piece, probabilities = predictor.predict_next(current_sequence[-6:])
        predicted_pieces.append(piece)
        current_sequence.append(piece)
    
    # Group predicted pieces into bags
    predicted_bags = []
    for i in range(0, len(predicted_pieces), 7):
        predicted_bags.append(predicted_pieces[i:i+7])
    
    return predicted_bags

def main():
    # Generate training data
    bag100 = generate_n_bags(100)
    bag20 = generate_n_bags(20)
    
    # Train the model
    predictor = train_lstm_model(n_bags=100, epochs=30, batch_size=32, save_model=True)
    
    # Predict next 20 bags
    predicted_bags = predict_next_bags(predictor, bag100, n_bags=20)
    
    # Display predictions
    print(f"\nPredicted {len(predicted_bags)} bags:")
    for i, bag in enumerate(predicted_bags, 1):
        print(f"Bag {i}: {bag}")
    
    # Create comparison dataframe
    flat_predicted = [piece for bag in predicted_bags for piece in bag]
    flat_actual = [piece for bag in bag20 for piece in bag]
    
    comparison_df = pd.DataFrame({
        'predicted': flat_predicted,
        'actual': flat_actual
    })
    
    # Add accuracy column
    comparison_df['correct'] = comparison_df['predicted'] == comparison_df['actual']
    
    print(f"\nPrediction Accuracy: {comparison_df['correct'].mean() * 100:.2f}%")
    print("\nComparison DataFrame:")
    print(comparison_df)
    
    return predictor, predicted_bags, comparison_df

if __name__ == "__main__":
    predictor, predicted_bags, comparison_df = main()