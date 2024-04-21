import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

file_path = 'Cleaned_Airbnb_Data2.csv'
cleaned_data = pd.read_csv(file_path, low_memory=False)

#list of amenities
amenities_list = ['', ' smooth pathway to front door', '24-hour check-in', 'Accessible-height bed', 'Accessible-height toilet',
                  'Air conditioning', 'Air purifier', 'BBQ grill', 'Baby bath', 'Baby monitor', 'Babysitter recommendations',
                  'Bath towel', 'Bathtub', 'Bathtub with shower chair', 'Beach essentials', 'Beachfront', 'Bed linens', 'Body soap',
                  'Breakfast', 'Buzzer/wireless intercom', 'Cable TV', 'Carbon monoxide detector', 'Cat(s)', 'Changing table',
                  'Children’s books and toys', 'Children’s dinnerware', 'Cleaning before checkout', 'Coffee maker', 'Cooking basics',
                  'Crib', 'Disabled parking spot', 'Dishes and silverware', 'Dishwasher', 'Dog(s)', 'Doorman', 'Doorman Entry',
                  'Dryer', 'EV charger', 'Elevator', 'Elevator in building', 'Essentials', 'Ethernet connection',
                  'Extra pillows and blankets', 'Family/kid friendly', 'Fire extinguisher', 'Fireplace guards', 'Firm mattress',
                  'First aid kit', 'Fixed grab bars for shower & toilet', 'Flat', 'Flat smooth pathway to front door',
                  'Free parking on premises', 'Free parking on street', 'Game console', 'Garden or backyard',
                  'Grab-rails for shower and toilet', 'Ground floor access', 'Gym', 'Hair dryer', 'Hand or paper towel', 'Hand soap',
                  'Handheld shower head', 'Hangers', 'Heating', 'High chair', 'Host greets you', 'Hot tub', 'Hot water',
                  'Hot water kettle', 'Indoor fireplace', 'Internet', 'Iron', 'Keypad', 'Kitchen', 'Lake access',
                  'Laptop friendly workspace', 'Lock on bedroom door', 'Lockbox', 'Long term stays allowed',
                  'Luggage dropoff allowed', 'Microwave', 'Other', 'Other pet(s)', 'Outlet covers', 'Oven',
                  'Pack ’n Play/travel crib', 'Paid parking off premises', 'Path to entrance lit at night', 'Patio or balcony',
                  'Pets allowed', 'Pets live on this property', 'Pocket wifi', 'Pool', 'Private bathroom', 'Private entrance',
                  'Private living room', 'Refrigerator', 'Roll-in shower with chair', 'Room-darkening shades', 'Safety card',
                  'Self Check-In', 'Shampoo', 'Single level home', 'Ski in/Ski out', 'Smart lock', 'Smoke detector',
                  'Smoking allowed', 'Stair gates', 'Step-free access', 'Stove', 'Suitable for events', 'TV', 'Table corner guards',
                  'Toilet paper', 'Washer', 'Washer / Dryer', 'Waterfront', 'Well-lit path to entrance', 'Wheelchair accessible',
                  'Wide clearance to bed', 'Wide clearance to shower & toilet', 'Wide clearance to shower and toilet',
                  'Wide doorway', 'Wide entryway', 'Wide hallway clearance', 'Window guards', 'Wireless Internet']

# Initializing the MultiLabelBinarizer with the full list of amenities
mlb = MultiLabelBinarizer()
mlb.classes_ = amenities_list

#filtering columns to include only the encoded amenities and other relevant data for modeling
X = cleaned_data[[col for col in cleaned_data.columns if col in mlb.classes_]]
y = cleaned_data['review_score_category']

# Split the data into training and testing sets--80%, 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#RandomForest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#retrieving feature importance
importances = rf.feature_importances_
feature_importances = pd.DataFrame(sorted(zip(importances, X.columns), reverse=True), columns=['Importance', 'Feature'])

#displaying top 10
print(feature_importances.head(10))
