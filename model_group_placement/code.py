from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('CollegePlacement.csv')

target = 'Placement'
features = df.columns.drop(target)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print("Training Set shape:", train_df.shape)
print("Testing Set shape:", test_df.shape)


def train_evaluate_model(model, name):
    print("Model:", name)
    # Fit the model
    model.fit(X_train, y_train)
    # Make prediction
    y_pred = model.predict(X_test)
    # Model evaluation
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    print(f"Confusion Matrix:\n {conf_matrix}\nAccuracy: {accuracy} "
          f"\nPrecision: {precision} \nRecall {recall} \nF1-Score {f1score}")
    # Return a dictionary
    return {"Model": name, "Accuracy": accuracy, "Precision": precision,
            "Recall": recall, "F1-Score": f1score}