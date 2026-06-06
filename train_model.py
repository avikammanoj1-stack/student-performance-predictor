import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("student_data.csv")

# Features
X = data[["StudyHours", "Attendance", "PreviousMarks", "Assignments"]]

# Target
y = data["FinalScore"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy metric
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))

# Test prediction
sample = [[6, 90, 80, 8]]
result = model.predict(sample)

print("Predicted Score:", round(result[0], 2))