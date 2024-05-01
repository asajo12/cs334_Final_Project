import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import top10Amenities as top

file_path = 'Airbnb_Data.csv'
airbnb_data = pd.read_csv(file_path)

# Dropping rows where 'host_since' or 'review_scores_rating' is missing
cleaned_data = airbnb_data.dropna(subset=['host_since', 'review_scores_rating'])

# Imputing missing values for bathrooms, bedrooms,beds with the median of each column
for col in ['bathrooms', 'bedrooms', 'beds']:
    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())

# Converting amenities from string to list of amenities
cleaned_data['amenities'] = cleaned_data['amenities'].apply(lambda x: set(x.strip('{}').replace('"', '').split(',')))

# One-hot encoding amenities
mlb = MultiLabelBinarizer()
amenities_encoded = mlb.fit_transform(cleaned_data['amenities'])
amenities_df = pd.DataFrame(amenities_encoded, columns=mlb.classes_, index=cleaned_data.index)

# Adding the encoded amenities back to the main dataframe 
cleaned_data = pd.concat([cleaned_data.reset_index(drop=True), amenities_df.reset_index(drop=True)], axis=1)

# Removing the original 'amenities' column as it's now encoded
cleaned_data = cleaned_data.drop('amenities', axis=1)

# Categorize review scores into high, mid, low
bins = [0, 60, 80, 100]
labels = ['low', 'mid', 'high']
cleaned_data['review_score_category'] = pd.cut(cleaned_data['review_scores_rating'], bins=bins, labels=labels, include_lowest=True)
top_amenities  = top.top10amenities (cleaned_data)['Feature'].tolist()
selected_columns = top_amenities + ['review_score_category', 'log_price']
cleaned_data = cleaned_data[selected_columns]

# Save the cleaned data to a new CSV file
cleaned_data.to_csv('Cleaned_Airbnb_Final.csv', index=False)

# Displaying first few rows
print(cleaned_data.info())
print(cleaned_data.head())




