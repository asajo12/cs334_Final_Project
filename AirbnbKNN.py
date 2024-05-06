import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv('Cleaned_Airbnb_Final20.csv', low_memory=False)

# Select x & y
X = data.drop ('review_score_category', axis = 1)
y = data['review_score_category'].astype('category')
#including price
X['log_price'] = data['log_price']


# Splitting the dataset 73/27
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'knn__n_neighbors': [1, 3, 5, 7, 10, 13, 15],  # Adjust values as needed
    'knn__metric': ['euclidean', 'manhattan', 'hamming']
}

# Create the pipeline with SMOTE, scaler, and KNN classifier
pipeline = ImblearnPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
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

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)