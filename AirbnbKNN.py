import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('Cleaned_Airbnb_Data_Updated.csv', low_memory=False)

# converting t/f to 1/0
bool_cols = ['host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
for col in bool_cols:
    data[col] = data[col].map({'t': 1, 'f': 0}).astype(float)  # Use float to avoid FutureWarning on downcasting

#dropiing cols that have text to minimize txt based noise
drop_columns = ['description', 'name', 'thumbnail_url', 'first_review', 'last_review', 'host_since']
data.drop(drop_columns, axis=1, inplace=True, errors='ignore')

# Define target and feats
X = data.drop('review_score_category', axis=1)
y = data['review_score_category'].astype('category')

# Handling percentages
if 'host_response_rate' in data.columns:
    data['host_response_rate'] = data['host_response_rate'].str.rstrip('%').astype('float') / 100

# Identifying categorical and numeric columns for transformation
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Pipeline setup
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
])

# Splitting the dataset 73/27
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

# Training
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
