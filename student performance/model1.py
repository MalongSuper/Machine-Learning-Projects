# Machine Learning Project 1: Predict Student Performance
# This project involves analyzing student performance data 
# to predict outcomes based on various features.
# Two subjects are considered: Mathematics and Portuguese language.
# Import the required libraries
# Train with Decision Tree Regressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Read the datasets
math_df = pd.read_csv('student/student-mat.csv', sep=';')
por_df = pd.read_csv('student/student-por.csv', sep=';')

# Print the first 5 rows of the both datasets
print("Mathematics Data:\n", math_df.head(5))
print("Portugese Language Data:\n", por_df.head(5))

# Encode categorical variables using one-hot encoding
# Replace True/False with 1/0 for binary columns
math_df_encoded = pd.get_dummies(math_df, drop_first=True).astype(int)
por_df_encoded = pd.get_dummies(por_df, drop_first=True).astype(int)
print("Encoded Mathematics Data:\n", math_df_encoded.head(5))
print("Encoded Portugese Language Data:\n", por_df_encoded.head(5))

# Split the data into training and testing sets (80% train, 20% test)
math_train, math_test = train_test_split(math_df_encoded, test_size=0.2, random_state=42)
por_train, por_test = train_test_split(por_df_encoded, test_size=0.2, random_state=42)

# Save them to CSV files
math_train.to_csv('math_train.csv', index=False)
math_test.to_csv('math_test.csv', index=False)
por_train.to_csv('por_train.csv', index=False)
por_test.to_csv('por_test.csv', index=False)

# Features and Target variables
# For mathematic dataset, predict final grade 'G3'
# Features are all columns except 'G1, 'G2', and 'G3'
math_features = math_df_encoded.columns.drop(['G1', 'G2', 'G3'])
X_math_train = math_train[math_features]
y_math_train = math_train['G3']
X_math_test = math_test[math_features]
y_math_test = math_test['G3']

# For portuguese dataset, predict final grade 'G3'
por_features = por_df_encoded.columns.drop(['G1', 'G2', 'G3'])
X_por_train = por_train[por_features]
y_por_train = por_train['G3']
X_por_test = por_test[por_features]
y_por_test = por_test['G3']

# Train the Mathematics model with Decision Tree Regression
math_model = DecisionTreeRegressor(random_state=42)
math_model.fit(X_math_train, y_math_train)

# Train the Portuguese model with Decision Tree Regression
por_model = DecisionTreeRegressor(random_state=42)
por_model.fit(X_por_train, y_por_train)

# Testing for mathematics dataset
math_y_pred = math_model.predict(X_math_test)
# Calculate metrics for mathematics model
math_mse = mean_squared_error(y_math_test, math_y_pred)
math_r2 = r2_score(y_math_test, math_y_pred)
print("Mathematics Model Performance:")
print(f"Mean Squared Error: {math_mse}")
print(f"R^2 Score: {math_r2}")


# Testing for portuguese dataset
por_y_pred = por_model.predict(X_por_test)
# Calculate metrics for portuguese model
por_mse = mean_squared_error(y_por_test, por_y_pred)
por_r2 = r2_score(y_por_test, por_y_pred)
print("Portuguese Model Performance:")
print(f"Mean Squared Error: {por_mse}")
print(f"R^2 Score: {por_r2}")


# Use a sample from the test set for demonstration
math_sample = X_math_test.iloc[0]  # Selecting any testing sample
print("Mathematics Sample data for testing: \n", math_sample.tolist())
# Predict
predicted_math_g3 = math_model.predict([math_sample])
print(f"Predicted G3 (Mathematics): {predicted_math_g3[0]}")
print(f"Actual G3 (Mathematics): {y_math_test.iloc[0]}")

por_sample = X_por_test.iloc[0]  # Selecting any testing sample
print("Portuguese Sample data for testing: \n", por_sample.tolist())
# Predict
predicted_por_g3 = por_model.predict([por_sample])
print(f"Predicted G3 (Portuguese): {predicted_por_g3[0]}")
print(f"Actual G3 (Portuguese): {y_por_test.iloc[0]}")


# The optional experiments that keep 'G1' and 'G2' as features have been
# moved to `Model2.py`. Run that file if you want to reproduce or extend
# the G1/G2 experiments separately.


