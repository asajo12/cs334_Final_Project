import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.impute import SimpleImputer



data = pd.read_csv('Cleaned_Airbnb_Data_Updated.csv')
amenity_features = [col for col in data.columns if 'amenity_' in col]
X = data[amenity_features]
y = data['review_score_category'].astype('category').cat.codes  # Convert categories to integer codes

# Imputing missing values
imputer = SimpleImputer(strategy='constant', fill_value=0)
X = imputer.fit_transform(X)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Converting labels to categorical
y = to_categorical(y)

# Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)


# NN model
model = Sequential()
model.add(Dense(64, input_dim=len(amenity_features), activation='relu'))  # Hidden layer
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))  # Output layer

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")


