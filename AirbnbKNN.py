import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv('Cleaned_Airbnb_Data_Updated.csv', low_memory=False)

# Converting 't'/'f' to 1/0 for boolean columns
bool_cols = ['host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
for col in bool_cols:
    if col in data.columns:
        data[col] = data[col].map({'t': 1, 'f': 0}).astype(float)

# Dropping cols that have text to minimize text-based noise
drop_columns = ['description', 'name', 'thumbnail_url', 'first_review', 'last_review', 'host_since']
data.drop(columns=drop_columns, inplace=True, errors='ignore')

# Focus on amenities columns; adjust the naming pattern as needed based on your actual data column names
amenity_features = [col for col in data.columns if 'amenity' in col]  # Example pattern
X = data[amenity_features]
y = data['review_score_category'].astype('category')

# Splitting the dataset 73/27
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

# Handling class imbalance with SMOTE and scaling features
pipeline = ImblearnPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
])

# Training
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
