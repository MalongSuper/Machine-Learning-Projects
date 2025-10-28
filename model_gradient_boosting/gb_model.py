import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('personality_dataset_extra.csv')

# Replacing categorical variables
object_columns = df.select_dtypes(include=['object']).columns
for i in object_columns:
    unique_values = df[i].unique()
    df[i] = df[i].map({unique_values[value]: value for value in range(len(unique_values))})
# Save this dataset
df.to_csv('new_dataset.csv', index=False)

# Split the data into 80% for training and 20% for testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# Features and Target
target = 'Personality'
features = df.columns.drop(target)
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Train the model with Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")
# Classification Report: Precision, Recall, F1-Score, Support
classification_report = classification_report(y_test, y_pred)
print(f"Classification Report: \n {classification_report}")

