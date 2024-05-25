import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(file_path):
    # TODO: Load processed data from CSV file
    df = pd.read_csv(file_path)
    return df

def split_data(df):
    # TODO: Split data into training and validation sets (the test set is already provided in data/test_data.csv)
    X = df.drop('Insect', axis = 1)
    y = df['Insect']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train, X_val, y_val):
    # TODO: Initialize your model and train it
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
    return model, val_score

def save_model(model, model_path):
    # TODO: Save your trained model
    joblib.dump(model, model_path)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df)
    model, val_score = train_model(X_train, y_train,X_val,y_val)
    print(val_score)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)