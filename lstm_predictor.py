import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tetramino_generator import generate_n_bags
import random

random.seed(42)
torch.manual_seed(42)

class TetrominoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TetrominoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=7, dropout=0.1):
        super(TetrominoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class TetrominoLSTMPredictor:
    def __init__(self, window_size=6):
        self.window_size = window_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.tetromino_shapes = ["I", "J", "L", "O", "S", "T", "Z"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, n_bags=1000):
        sequences = generate_n_bags(n_bags)
        flat_sequence = [piece for bag in sequences for piece in bag]
        
        encoded_sequence = self.label_encoder.fit_transform(flat_sequence)
        
        X, y = [], []
        for i in range(len(encoded_sequence) - self.window_size):
            X.append(encoded_sequence[i:i + self.window_size])
            y.append(encoded_sequence[i + self.window_size])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y
    
    def build_model(self):
        self.model = TetrominoLSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=len(self.tetromino_shapes),
            dropout=0.1
        ).to(self.device)
        return self.model
    
    def train(self, n_bags=1000, epochs=50, batch_size=32, validation_split=0.2):
        X, y = self.prepare_data(n_bags)
        
        if self.model is None:
            self.build_model()
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        train_dataset = TetrominoDataset(X_train, y_train)
        val_dataset = TetrominoDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {100*train_correct/train_total:.2f}%, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, '
                  f'Val Acc: {100*val_correct/val_total:.2f}%')
    
    def predict_next(self, last_6_pieces):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        encoded_pieces = self.label_encoder.transform(last_6_pieces)
        
        X = np.array(encoded_pieces, dtype=np.float32).reshape(1, self.window_size, 1)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
            probabilities = probabilities.cpu().numpy()[0]
        
        predicted_piece = self.label_encoder.inverse_transform([predicted_index])[0]
        
        return predicted_piece, probabilities
    
    def predict_next_n(self, last_6_pieces, n=5):
        predictions = []
        current_sequence = list(last_6_pieces)
        
        for _ in range(n):
            piece, probabilities = self.predict_next(current_sequence[-self.window_size:])
            predictions.append((piece, probabilities))
            current_sequence.append(piece)
        
        return predictions
    
    def save_model(self, filepath='lstm_tetromino_model.pth'):
        if self.model is None:
            raise ValueError("No model to save.")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder_classes': self.label_encoder.classes_,
            'window_size': self.window_size
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='lstm_tetromino_model.pth'):
        checkpoint = torch.load(filepath)
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_encoder.classes_ = checkpoint['label_encoder_classes']
        self.window_size = checkpoint['window_size']
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")


def main():
    """Example usage"""
    predictor = TetrominoLSTMPredictor(window_size=6)
    
    # Train the model
    print("Training LSTM model...")
    history = predictor.train(n_bags=500, epochs=30, batch_size=32)
    
    # Save the model
    predictor.save_model()
    
    # Test prediction
    test_sequence = ["I", "J", "L", "O", "S", "T"]
    predicted_piece, probabilities = predictor.predict_next(test_sequence)
    
    print(f"\nTest sequence: {test_sequence}")
    print(f"Predicted next piece: {predicted_piece}")
    print(f"Probabilities:")
    for piece, prob in zip(predictor.tetromino_shapes, probabilities):
        print(f"  {piece}: {prob:.4f}")
    
    # Predict next 5 pieces
    print("\nPredicting next 5 pieces:")
    predictions = predictor.predict_next_n(test_sequence, n=5)
    for i, (piece, probs) in enumerate(predictions, 1):
        print(f"  Piece {i}: {piece}")


if __name__ == "__main__":
    main()
