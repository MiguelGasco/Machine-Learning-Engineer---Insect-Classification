import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # TODO: Load data from CSV file
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    # TODO: Handle missing values, outliers, etc.
    z_scores  = df.select_dtypes(include=['int64', 'float64']).drop('Insect', axis=1)
    outliers = df[(z_scores < 3).all(axis=1)]
    
    if len(outliers) > 0: 
        df_clean = df[~df.index.isin(outliers.index)] 
    else:
        print("No se encontraron outliers.")
        df_clean = df
        
    return df_clean

def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.
    target = df['Insect']
    features = df.drop('Insect', axis=1)
    
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)
    
    df_processed = pd.concat([pd.DataFrame(features_standardized, columns=features.columns), target], axis=1)
    
    return df_processed

def save_data(df_train, output_file):
    # TODO: Save processed data to a CSV file
    df_train.to_csv(output_file, index=False)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/train.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean_train= clean_data(df)
    df_processed = preprocess_data(df_clean_train)
    save_data(df_processed, output_file)
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)