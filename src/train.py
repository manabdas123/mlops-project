import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv("data/Titanic-Dataset.csv")

# Preprocessing
df = df[['Pclass','Sex','Age','Fare','Survived']]
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

import joblib
joblib.dump(model, "model.pkl")

# MLflow tracking
mlflow.set_experiment("Titanic")

with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    