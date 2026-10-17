
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("Advertising.csv")
df.columns = df.columns.str.strip()

print("DATASET INFORMATION")
print("=" * 40)
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

features = ["TV", "Radio", "Newspaper"]
target = "Sales"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nSALES PREDICTION RESULTS")
print("=" * 40)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Mean Absolute Error: {mae:.3f}")
print(f"Root Mean Squared Error: {rmse:.3f}")
print(f"R-squared Score: {r2:.3f}")

results = pd.DataFrame({
    "Actual Sales": y_test.to_numpy(),
    "Predicted Sales": predictions
})
results.to_csv("outputs/prediction_results.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())
plt.plot([minimum, maximum], [minimum, maximum], "r--")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(features, model.coef_)
plt.xlabel("Advertising Channel")
plt.ylabel("Regression Coefficient")
plt.title("Advertising Features and Predicted Sales")
plt.tight_layout()
plt.savefig("outputs/advertising_coefficients.png")
plt.close()

importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})
importance.to_csv("outputs/advertising_coefficients.csv", index=False)

print("\nOutput files saved in the outputs folder.")
print("Project completed successfully.")
