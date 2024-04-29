import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load data
try:
    df = pd.read_csv('/home/kodex/data/Combined_News_DJIA(train).csv')
    reddit_news = pd.read_csv('/home/kodex/data/Test_Combined_News.csv')
except FileNotFoundError as e:
    print("File not found:", e)

try:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
except KeyError as e:
    print("KeyError:", e)

# Merge and preprocess data
merged_data = df.drop(columns=['Date']).dropna()

# Prepare sequences
seq_length = 10

def prepare_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Prepare train and validation data
train_data, val_data = train_test_split(merged_data.to_numpy(), test_size=0.2, shuffle=False)

X_train = prepare_sequences(train_data, seq_length)
y_train = train_data[seq_length:, -1]

X_val = prepare_sequences(val_data, seq_length)
y_val = val_data[seq_length:, -1]

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Initialize model, loss function, and optimizer
input_size = merged_data.shape[1]
hidden_size = 128
output_size = 1
num_layers = 1

lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    lstm_model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = lstm_model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    # Validation
    with torch.no_grad():
        lstm_model.eval()
        val_outputs = lstm_model(X_val_tensor)
        val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss.item()}")

# Make predictions
with torch.no_grad():
    lstm_model.eval()
    lstm_predictions = lstm_model(X_val_tensor)

# Inverse scaling
scaler = MinMaxScaler()
scaler.fit(merged_data)
predicted_close_prices = lstm_predictions.squeeze().numpy()
inverse_transformed_predictions = np.hstack((np.zeros((lstm_predictions.shape[0], merged_data.shape[1] - 1)), predicted_close_prices.reshape(-1, 1)))
lstm_predictions = scaler.inverse_transform(inverse_transformed_predictions)

# Calculate MSE
mse = mean_squared_error(df[-len(predicted_close_prices):]['Close'], predicted_close_prices)
print("Model MSE:", mse)

# Save predictions to submission file
submission_df = pd.DataFrame({
    'Id': range(1, len(predicted_close_prices) + 1),
    'Close': predicted_close_prices
})

submission_df.to_csv("submission.csv", index=False)
