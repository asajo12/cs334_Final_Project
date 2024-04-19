#Step 1: want to preprocess the data 
    # get rid of missing values if applicable 
    #categorize numerical and categorical values 
    # one-hot-encoding or min-max when needed 
    
#import argparse
import pandas as pd
#import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def handle_missing_vals(data): #ngl idk if this is the right approach, we could do a diff way but lmk.
    #handling missing values by dropping those columns
    columns_dropped = data.columns[data.isna().any()].tolist() #making a list so we can print for testing purposes

    cleaned_data = data.dropna(axis=1) #print to see if the cols are dropped
    
    print("COLUMNS DROPPED:", columns_dropped)
    return cleaned_data

def load_dataset(file):
    return pd.read_csv(file)

def main():
    file_path = 'Airbnb_Data.csv'
    data = load_dataset(file_path)
    
    cleaned_data = handle_missing_vals(data) 
    
    #checking if the data loading step is complete
    print("-----------------------------------------------------------")
    print("Dataset loaded successfully.")
    print("ORIGINAL DATASET INFORMATION: ")
    print("Number of rows:", data.shape[0])
    print("Number of columns:", data.shape[1])
    print("\nColumns in the dataset:")
    print(data.columns)
    print("\nFirst few rows of the dataset:")
    print(data.head())
    print("-----------------------------------------------------------")
    
    
    #checking if the data is cleaned (missing values are dropped)
    print("-----------------------------------------------------------")
    print("CLEANED DATASET INFORMATION:")
    print("Number of rows:", cleaned_data.shape[0])
    print("Number of columns:", cleaned_data.shape[1])
    print("\nColumns in the cleaned dataset:")
    print(cleaned_data.columns)
    print("\nFirst few rows of the cleaned dataset:")
    print(cleaned_data.head())
    print("-----------------------------------------------------------")
    
if __name__ == "__main__":
    main()