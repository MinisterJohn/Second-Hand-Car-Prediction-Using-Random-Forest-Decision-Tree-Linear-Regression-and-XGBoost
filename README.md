# Second-Hand-Car-Prediction-Using-Random-Forest-Decision-Tree-Linear-Regression-and-XGBoost
This model will be able to predict the price of second hand cars.

Original file is located at
    https://colab.research.google.com/drive/143O1wwGBTCErjzlsx86dlqiKLD3vxoE_

**# Step 1: Import the necessary libraries**

1. import pandas as pd
2. import numpy as np
3. from sklearn.model_selection import train_test_split
4. from sklearn.ensemble import RandomForestRegressor
5. from sklearn.neural_network import MLPRegressor
6. from sklearn.metrics import mean_squared_error
7. from sklearn.preprocessing import StandardScaler
8. from tensorflow.keras.models import Sequential
9. from tensorflow.keras.layers import Dense

**# Step 2: Load and preprocess the dataset**
df = pd.read_csv("/content/train-data.csv")
df.head()



"""Step 3: Select the columns that you need"""

selected_features = ['Year', 'Kilometers_Driven',  'Transmission', 'Owner_Type', 'Engine', 'Power', 'Seats', 'Price']



## Create a mapping dictionary to replace categorical values with numerical values
Transmission = {'Manual': 0, 'Automatic': 1}

# Replace the values in the 'Tranmission' column with the numerical representation
df['Transmission'] = df['Transmission'].replace(Transmission)

## Create a mapping dictionary to replace categorical values with numerical values
Owner_Type = {'First': 0, 'Second': 1, 'Third': 2, 'Fourth & Above': 3}

# Replace the values in the 'Tranmission' column with the numerical representation
df['Owner_Type'] = df['Owner_Type'].replace(Owner_Type)

#Create a dataframe with the selected columns
new_dataset = df[selected_features]

new_dataset

"""# **Preprocessing stage**
This code finds the mode of the 'Seats' Colum and then fill the missing values in the 'Seats' Column with the mode value
"""

import pandas as pd


# Find the mode of the 'Seats' column
mode_seats = new_dataset['Seats'].mode()[0]

# Print the mode value
print("Mode of 'Seats' column:", mode_seats)

# Fill the missing values in 'Seats' column with the mode value
new_dataset['Seats'].fillna(mode_seats, inplace=True)

# Now, the missing values in the 'Seats' column have been replaced with the mode value.

# Verify that there are no more missing values in the 'Seats' column
print("Null Values in the 'Seats' Column After Filling:")
print(new_dataset['Seats'].isnull().sum())

"""# **Preprocessing stage**
This code remove 'bhp' from the 'Power' colum and then converts it to a numeric data type.
"""

import pandas as pd

# Remove 'bhp' from the 'Power' column
new_dataset['Power'] = df['Power'].str.replace(' bhp', '', regex=True)

# Now, the 'Power' column contains numeric values without 'bhp'.

# Convert the 'Power' column to numeric dtype (float or int)
new_dataset['Power'] = pd.to_numeric(new_dataset['Power'], errors='coerce')

# 'coerce' argument will convert non-numeric values (e.g., NaN) to NaN.

# Verify the 'Power' column after the transformation
print(new_dataset['Power'])

"""# **Preprocessing stage**
This code finds the mode of the 'Power' Colum and then fill the missing values in the 'Power' Column with the mode value
"""

import pandas as pd


# Find the mode of the 'Seats' column
mode_power = new_dataset['Power'].mode()[0]

# Print the mode value
print("Mode of 'Power' column:", mode_power)

# Fill the missing values in 'Seats' column with the mode value
new_dataset['Power'].fillna(mode_power, inplace=True)

# Now, the missing values in the 'Seats' column have been replaced with the mode value.

# Verify that there are no more missing values in the 'Seats' column
print("Null Values in the 'Power' Column After Filling:")
print(new_dataset['Power'].isnull().sum())

import pandas as pd

# Remove 'CC' from the 'Engine' column
new_dataset['Engine'] = df['Engine'].str.replace(' CC', '', regex=True)

# Now, the 'Engine' column contains numeric values without 'bhp'.

# Convert the 'Engine' column to numeric dtype (float or int)
new_dataset['Engine'] = pd.to_numeric(new_dataset['Engine'], errors='coerce')

# 'coerce' argument will convert non-numeric values (e.g., NaN) to NaN.

# Verify the 'Power' column after the transformation
print(new_dataset['Engine'])

"""# **Preprocessing**
The code bleow finds the mode of the Engine column, prints the mode value, and then fill the missing values
"""

import pandas as pd


# Find the mode of the 'Engine' column
mode_engine = new_dataset['Engine'].mode()[0]

# Print the mode value
print("Mode of 'Engine' column:", mode_engine)

# Fill the missing values in 'Egine' column with the mode value
new_dataset['Engine'].fillna(mode_engine, inplace=True)

# Now, the missing values in the 'Engine' column have been replaced with the mode value.

# Verify that there are no more missing values in the 'Seats' column
print("Null Values in the 'Engine' Column After Filling:")
print(new_dataset['Engine'].isnull().sum())

#Check for null values in the entire DataFrame
null_values = new_dataset.isnull().sum()

null_values

new_dataset

# Split the dataset into features (X) and target variable (y)
X = new_dataset.drop('Price', axis =1)
y = new_dataset['Price']

X

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_train

# Drop rows with missing values in X_train and y_train
X_train.dropna(inplace=True)
y_train = y_train[X_train.index]

y_train

from sklearn.impute import SimpleImputer

# Create an imputer instance
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on X_train
X_train_imputed = imputer.fit_transform(X_train)

# Convert the imputed data back to a DataFrame
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)

# Drop rows with missing values in X_test and y_test
X_test.dropna(inplace=True)
y_test = y_test[X_test.index]

# Assuming X_test is a DataFrame containing the test features
# Adjust the DataFrame name and column names accordingly

X_test.drop('Mileage', axis=1, inplace=True)

# Display the updated X_test DataFrame
print(X_test)

from sklearn.impute import SimpleImputer

# Create an imputer instance
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on X_train
X_test_imputed = imputer.fit_transform(X_test)

# Convert the imputed data back to a DataFrame
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# Scale the features using StandardScaler
# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

X_test

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# Step 7: Train and Evaluate the Random Forest Regressor

rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
print("Random Forest Mean Squared Error:", rf_mse)
rf_predictions = rf_model.predict(X_test)
rf_r2_score = r2_score(y_test, rf_predictions)
print("Random Forest R2_score:", rf_r2_score)

**Output:** 
Random Forest Mean Squared Error: **16.828281614758023**
Random Forest R2_score: **0.863250844320417**


# Step 8:  Training the Decision Tree Regressor 
from sklearn.tree import DecisionTreeRegressor
# Create a Decision Tree Regressor
decision_tree = DecisionTreeRegressor()
# Train the model
decision_tree.fit(X_train, y_train)

# Make predictions on the test data
dt_predictions = decision_tree.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
print("Decision Tree Mean Squared Error: ", dt_mse)
dt_r2score = r2_score(y_test, dt_predictions)
print("Decision Tree R2 Score: ", dt_r2score)
**Output:** 
Decision Tree Mean Squared Error:  **33.25287157518013**
Decision Tree R2 Score: ** 0.7325448732590114**

**Step 9: Training the Linear Regression model** 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Linear Regression model
linear_regression = LinearRegression()

# Train the model
linear_regression.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
linear_predictions = linear_regression.predict(X_test_scaled)
lr_mse = mean_squared_error(y_test, linear_predictions)
print("Decision Tree Mean Squared Error: ", lr_mse)
lr_r2score = r2_score(y_test, linear_predictions)
print("Decision Tree R2 Score: ", lr_r2score)
**Output: **


!pip install xgboost

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Create a DMatrix for the training and test data
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Define the XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Create the XGBoost model
xgb_model = xgb.train(params, dtrain)

# Make predictions on the test data
xgb_predictions = xgb_model.predict(dtest)
# Calculate the mean squared error (MSE)
mse = mean_squared_error(y_test, xgb_predictions)
print(f"Mean Squared Error: {mse}")

r2score = r2_score(y_test, xgb_predictions)
print(f"R2 Score: {r2score}")

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


# Calculate the R-squared (R2) score
r2 = r2_score(y_test, svm_predictions)
print(f"R-squared (R2) Score: {r2}")

