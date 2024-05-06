import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('Cleaned_Airbnb_Final.csv')

X = data.drop(['review_score_category', 'log_price'], axis=1)
y = data['review_score_category']

class_counts = y.value_counts()
max_class_count = class_counts.max()

resampled_list = []

for class_index in class_counts.index:
    class_subset = data[data['review_score_category'] == class_index]
    resampled_class_subset = class_subset.sample(max_class_count, replace=True, random_state=42)
    resampled_list.append(resampled_class_subset)

resampled_data = pd.concat(resampled_list, ignore_index=True)
resampled_data = resampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

X_resampled = resampled_data.drop(['review_score_category', 'log_price'], axis=1)
y_resampled = resampled_data['review_score_category']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_encoded, test_size=0.27, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(max_iter=100, early_stopping=True, n_iter_no_change=10)
mlp.fit(X_train_scaled, y_train)

#class. report
y_pred = mlp.predict(X_test_scaled)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (150, 100)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01],
}

grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validated score:", grid_search.best_score_)

best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(X_test_scaled)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))