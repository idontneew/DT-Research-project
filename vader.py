import os
import json
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Initialize the VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Function to load and preprocess JSON data
def load_data_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    # Preprocess data if needed
    return data

# Function to perform sentiment analysis using VADER
def analyze_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']

# Function to split data into train and test sets
def split_data(data, test_size=0.2):
    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data

# Function to evaluate sentiment analysis model
def evaluate_model(test_data):
    true_sentiments = []
    predicted_sentiments = []
    for item in test_data:
        text = item['text']
        true_sentiment = item['sentiment']
        predicted_sentiment = analyze_sentiment(text)
        true_sentiments.append(true_sentiment)
        predicted_sentiments.append(predicted_sentiment)
    # Calculate evaluation metrics
    report = classification_report(true_sentiments, predicted_sentiments)
    print("Classification Report:\n", report)

# Path to the directory containing the JSON files (87 folders representing each company)
data_directory = "/home/kodex/preprocessed"

# Load data from JSON files
all_data = []
for folder_name in os.listdir(data_directory):
    folder_path = os.path.join(data_directory, folder_name)
    if os.path.isdir(folder_path):
        # Find JSON files in the folder
        json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
        for json_file in json_files:
            json_path = os.path.join(folder_path, json_file)
            data = load_data_from_json(json_path)
            all_data.extend(data)

# Split data into train and test sets
train_data, test_data = split_data(all_data)

# Evaluate the sentiment analysis model
evaluate_model(test_data)
