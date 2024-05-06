import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv('Cleaned_Airbnb_Final.csv')

X = data[['Family/kid friendly', 'Air conditioning', 'Carbon monoxide detector', 'TV', 
          'Lock on bedroom door', 'Fire extinguisher', 'Buzzer/wireless intercom', 
          'Free parking on premises', 'Laptop friendly workspace', 'Shampoo']]  
y = data['log_price'] # Target variable

regression_pipeline = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regression_pipeline.fit(X_train, y_train)

y_pred = regression_pipeline.predict(X_test)

# Evaluate the model
r2_score = metrics.r2_score(y_test, y_pred)
print('R2 Score:', r2_score)
