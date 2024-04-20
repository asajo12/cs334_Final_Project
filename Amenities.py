import pandas as pd

# Load your data (adjust the path as necessary)
file_path = 'Airbnb_Data.csv'
data = pd.read_csv(file_path, low_memory=False)

# Check if 'amenities' column exists
if 'amenities' in data.columns:
    # Convert amenities strings to sets
    def parse_amenities(amenities_str):
        return set(amenities_str.strip('{}').replace('"', '').split(','))

    data['amenities'] = data['amenities'].apply(parse_amenities)

    # Aggregate all unique amenities across all listings
    all_amenities = set()
    data['amenities'].apply(lambda x: all_amenities.update(x))

    # Convert the set to a sorted list to view them more easily
    sorted_amenities = sorted(all_amenities)

    # Optionally, print or inspect the amenities
    print(sorted_amenities)
else:
    print("Column 'amenities' does not exist in the dataset. Check the column names and ensure you have the correct dataset loaded.")
