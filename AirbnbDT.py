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

data = pd.read_csv('Cleaned_Airbnb_Final20.csv')


# Focus on amenities columns; adjust the naming pattern as needed based on your actual data column names
X = data.drop ('review_score_category', axis = 1)
y = data['review_score_category'].astype('category')

#including prices
X['log_price'] = data['log_price']

# Splitting the dataset 73/27
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

# Handling class imbalance with SMOTE and scaling features
pipeline = ImblearnPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('DT', RandomForestClassifier ())
    #('DT', DecisionTreeClassifier())
])

# Define the hyperparameter grid for Decision Tree Classifier
param_grid = {
    'DT__criterion': ['gini', 'entropy', 'log_loss'], 
    'DT__n_estimators': [50, 100, 200, 300],
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
