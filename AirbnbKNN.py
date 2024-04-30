import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.metrics import classification_report, accuracy_score

# for the graph
import seaborn as sns
import matplotlib.pyplot as plt

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
#including price
X['log_price'] = data['log_price']

# Splitting the dataset 73/27
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'knn__n_neighbors': [1, 3, 5, 7, 10, 13, 15],  # Adjust values as needed
    'knn__metric': ['euclidean', 'manhattan']
}

# Create the pipeline with SMOTE, scaler, and KNN classifier
pipeline = ImblearnPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier())
])

# Create GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring= 'accuracy')

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

correlation_matrix = X.corr()
#plot heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot = True, cmap="coolwarm", fmt=".2f", cbar=True, square=True)
plt.title('Correlation Heatmap of Amenities')
plt.show()

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)