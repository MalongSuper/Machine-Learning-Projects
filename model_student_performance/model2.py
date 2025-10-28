# Machine Learning Project 1: Predict Student Performance (with G1 & G2 as features)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Read the datasets
math_df = pd.read_csv('student/student-mat.csv', sep=';')
por_df = pd.read_csv('student/student-por.csv', sep=';')

# Print the first 5 rows of the both datasets
print("Mathematics Data:\n", math_df.head(5))
print("Portuguese Language Data:\n", por_df.head(5))

# Encode categorical variables using one-hot encoding
math_df_encoded = pd.get_dummies(math_df, drop_first=True).astype(int)
por_df_encoded = pd.get_dummies(por_df, drop_first=True).astype(int)

print("Encoded Mathematics Data:\n", math_df_encoded.head(5))
print("Encoded Portuguese Language Data:\n", por_df_encoded.head(5))

# Split the data into training and testing sets (80% train, 20% test)
math_train, math_test = train_test_split(math_df_encoded, test_size=0.2, random_state=42)
por_train, por_test = train_test_split(por_df_encoded, test_size=0.2, random_state=42)

# Save to CSV
math_train.to_csv('math_train.csv', index=False)
math_test.to_csv('math_test.csv', index=False)
por_train.to_csv('por_train.csv', index=False)
por_test.to_csv('por_test.csv', index=False)

# Features and Target variables (G1 and G2 are now INCLUDED)
math_features = math_df_encoded.columns.drop(['G3'])
X_math_train = math_train[math_features]
y_math_train = math_train['G3']
X_math_test = math_test[math_features]
y_math_test = math_test['G3']

por_features = por_df_encoded.columns.drop(['G3'])
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

# Evaluate Mathematics model
math_y_pred = math_model.predict(X_math_test)
math_mse = mean_squared_error(y_math_test, math_y_pred)
math_r2 = r2_score(y_math_test, math_y_pred)
print("\nMathematics Model Performance (with G1 & G2):")
print(f"Mean Squared Error: {math_mse}")
print(f"R^2 Score: {math_r2}")

# Evaluate Portuguese model
por_y_pred = por_model.predict(X_por_test)
por_mse = mean_squared_error(y_por_test, por_y_pred)
por_r2 = r2_score(y_por_test, por_y_pred)
print("\nPortuguese Model Performance (with G1 & G2):")
print(f"Mean Squared Error: {por_mse}")
print(f"R^2 Score: {por_r2}")

# Demonstration with a sample from Mathematics
math_sample = X_math_test.iloc[0]
predicted_math_g3 = math_model.predict([math_sample])
print("\nMathematics Sample Data for Prediction:\n", math_sample.tolist())
print(f"Predicted G3 (Mathematics): {predicted_math_g3[0]}")
print(f"Actual G3 (Mathematics): {y_math_test.iloc[0]}")

# Demonstration with a sample from Portuguese
por_sample = X_por_test.iloc[0]
predicted_por_g3 = por_model.predict([por_sample])
print("\nPortuguese Sample Data for Prediction:\n", por_sample.tolist())
print(f"Predicted G3 (Portuguese): {predicted_por_g3[0]}")
print(f"Actual G3 (Portuguese): {y_por_test.iloc[0]}")
