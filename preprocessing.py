import pandas as pd

def handle_missing_vals(data):
    # Filling missing values with zeros
    cleaned_data = data.fillna(0)
    return cleaned_data

def load_dataset(file):
    return pd.read_csv(file)

def main():
    file_path = 'Airbnb_Data.csv'
    data = load_dataset(file_path)
    
    cleaned_data = handle_missing_vals(data) 
    
    # Checking if the data loading step is complete
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
    
    
    # Checking if the data is cleaned (missing values are filled with zeros)
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
