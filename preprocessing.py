#Step 1: want to preprocess the data 
    # get rid of missing values if applicable 
    #categorize numerical and categorical values 
    # one-hot-encoding or min-max when needed 
    
#import argparse
import pandas as pd
#import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_dataset(file):
    return pd.read_csv(file)

def main():
    file_path = 'Airbnb_Data.csv'
    data = load_dataset(file_path)
    
    #checking if the data loading step is complete
    print("Dataset loaded successfully.")
    print("Number of rows:", data.shape[0])
    print("Number of columns:", data.shape[1])
    print("\nColumns in the dataset:")
    print(data.columns)
    print("\nFirst few rows of the dataset:")
    print(data.head())
    
if __name__ == "__main__":
    main()