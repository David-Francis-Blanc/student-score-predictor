import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("study_data.csv")

# Prepare features and target
X = data[["Hours"]]
y = data["Score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict for 7 hours
prediction = model.predict([[7]])
print(f"Predicted score for 7 hours: {prediction[0]:.2f}")

# Plot
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Study Hours vs Score (Linear Regression)")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.grid(True)
plt.show()
