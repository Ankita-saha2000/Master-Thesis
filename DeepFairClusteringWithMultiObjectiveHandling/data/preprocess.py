import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def save_processed_data_to_txt(data_path):
    # Load the synthetic Adult-like dataset
    synthetic_adult_data = pd.read_csv(data_path)

    # Step 1: Preprocess the data
    # One-hot encode categorical features
    categorical_features = ['Education', 'Occupation', 'Sex', 'Race']
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    categorical_transformed = one_hot_encoder.fit_transform(synthetic_adult_data[categorical_features])

    # Standardize numerical features
    numerical_features = ['Age', 'HoursPerWeek']
    scaler = StandardScaler()
    numerical_transformed = scaler.fit_transform(synthetic_adult_data[numerical_features])

    # Combine processed features
    processed_features = np.hstack([numerical_transformed, categorical_transformed])

    # Extract labels (e.g., 'Income')
    labels = synthetic_adult_data['Income'].values

    # Print the shape of the processed features
    print("Shape of processed features:", processed_features.shape)

    # Print the first few rows of processed features and labels
    print("First few rows of processed features:\n", processed_features[:5])
    print("First few labels:\n", labels[:5])

    # Save the processed features and labels as text files
    np.savetxt('dummy_processed_features.txt', processed_features)
    np.savetxt('dummy_labels.txt', labels, fmt='%d')

    print("Files 'processed_features.txt' and 'labels.txt' have been successfully saved.")

if __name__ == "__main__":
    # Replace 'path_to_your/synthetic_adult_data.csv' with the actual path to your CSV file
    save_processed_data_to_txt('dummy.csv')

