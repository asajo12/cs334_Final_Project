import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def handle_missing_vals(data):
    # Filling missing values with zeros
    cleaned_data = data.fillna(0)
    return cleaned_data

def load_dataset(file):
    return pd.read_csv(file)

def impute_missing_values(data):
    # Imputing missing values for 'bathrooms', 'bedrooms', 'beds' with the median of each column
    for col in ['bathrooms', 'bedrooms', 'beds']:
        data[col] = data[col].fillna(data[col].median())
    return data

def encode_amenities(data):
    # Converting amenities from string to list of amenities
    data['amenities'] = data['amenities'].apply(lambda x: set(x.strip('{}').replace('"', '').split(',')))

    # One-hot encoding amenities
    mlb = MultiLabelBinarizer()
    amenities_encoded = mlb.fit_transform(data['amenities'])
    amenities_df = pd.DataFrame(amenities_encoded, columns=mlb.classes_, index=data.index)

    # Adding the encoded amenities back to the main dataframe
    data = pd.concat([data.reset_index(drop=True), amenities_df.reset_index(drop=True)], axis=1)

    # Removing the original 'amenities' column as it's now encoded
    data = data.drop('amenities', axis=1)
    return data

def categorize_review_scores(data):
    # Categorize review scores into high, mid, low
    bins = [0, 60, 80, 100]
    labels = ['low', 'mid', 'high']
    data['review_score_category'] = pd.cut(data['review_scores_rating'], bins=bins, labels=labels, include_lowest=True)
    return data

def save_to_csv(data, filename):
    # Save the cleaned data to a new CSV file
    data.to_csv(filename, index=False)

def preprocess_data(file_path):
    data = load_dataset(file_path)
    cleaned_data = handle_missing_vals(data)
    cleaned_data = impute_missing_values(cleaned_data)
    cleaned_data = encode_amenities(cleaned_data)
    cleaned_data = categorize_review_scores(cleaned_data)
    return cleaned_data

def main():
    file_path = 'Airbnb_Data.csv'
    cleaned_data = preprocess_data(file_path)
    save_to_csv(cleaned_data, 'Cleaned_Airbnb_Data.csv')
    print(cleaned_data.info())
    print(cleaned_data.head())
    
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
