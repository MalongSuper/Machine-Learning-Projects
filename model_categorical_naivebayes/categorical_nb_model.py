import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import CategoricalNB

df = pd.read_csv('adult.csv')

# Replace the "?" in the features with the mode
for i in df.columns:
    df[i] = df[i].replace('?', df[i].mode().iloc[0])

# Create 4 groups for age
df['age'] = pd.cut(df['age'], bins=4,
                   labels=[f'Age_{i+1}' for i in range(4)])

# Create 8 groups for fnlwgt
df['fnlwgt'] = pd.cut(df['fnlwgt'], bins=8,
                      labels=[f'Group_{i+1}' for i in range(8)])

# Replacing categorical variables
object_columns = df.select_dtypes(include=['object', 'category']).columns
for i in object_columns:
    unique_values = df[i].unique()
    df[i] = df[i].map({unique_values[value]: value for value in range(len(unique_values))})
print(df.head(20))

# Training and Testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Features and Target
target = 'income'
features = df.columns.drop(target)
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Train the model with Naive Bayes Classifier
model = CategoricalNB(alpha=1)  # The alpha for smoothing
model.fit(X_train, y_train)

# Add Accuracy Score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")
# Classification Report: Precision, Recall, F1-Score, Support
classification_report = classification_report(y_test, y_pred)
print(f"Classification Report: \n {classification_report}")
