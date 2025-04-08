# Predict student scores based on study hours

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Study hours and scores
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([50, 55, 65, 70, 75])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict score for 6 hours of study
predicted = model.predict([[6]])
print(f"Predicted score for 6 hours of study: {predicted[0]:.2f}")

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Study Hours vs Score")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
