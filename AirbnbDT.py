import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.metrics import classification_report, accuracy_score

# for the graph
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Cleaned_Airbnb_Data_Updated.csv')

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

#including prices
X['log_price'] = data['log_price']

# Splitting the dataset 73/27
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

# Handling class imbalance with SMOTE and scaling features
pipeline = ImblearnPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    # ('DT', RandomForestClassifier ())
    ('DT', DecisionTreeClassifier())
])

# Define the hyperparameter grid for Decision Tree Classifier
param_grid = {
    'DT__criterion': ['gini', 'entropy'], 
    # 'DT__n_estimators': [50, 100, 200, 300],
    'DT__max_depth': [1, 2, 5, 10, 15, 50, 100],
    'DT__min_samples_leaf': [1, 2, 3, 5, 10, 15]
}

# Create GridSearchCV object with the pipeline and parameter grid
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv= 5, scoring = 'accuracy', verbose=1, n_jobs=-1)

# Perform grid search and hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Use the best model for prediction
y_pred = best_model.predict(X_test)

# Evaluation with best model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=1)

print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

correlation_matrix = X.corr()

#plot heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot = True, cmap="coolwarm", fmt=".2f", cbar=True, square=True)
plt.title('Correlation Heatmap of Amenities')
plt.show()

'''
# Training
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

'''