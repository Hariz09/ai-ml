# House Price Predictor - Your First Machine Learning Project
# This project teaches fundamental ML concepts through a practical example

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Create synthetic dataset (in real projects, you'd load actual data)
print("Step 1: Creating dataset...")
np.random.seed(42)  # For reproducible results

# Generate synthetic house data
n_samples = 1000
house_sizes = np.random.normal(2000, 500, n_samples)  # Square feet
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)  # Years old

# Create realistic price formula with some noise
prices = (house_sizes * 150 + bedrooms * 5000 + bathrooms * 8000 - age * 1000 + 
          np.random.normal(0, 20000, n_samples))

# Create DataFrame
data = pd.DataFrame({
    'size': house_sizes,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'price': prices
})

print(f"Dataset created with {len(data)} houses")
print("\nFirst 5 rows:")
print(data.head())
print("\nDataset statistics:")
print(data.describe())

# Step 2: Data visualization
print("\nStep 2: Visualizing the data...")
plt.figure(figsize=(12, 8))

# Plot 1: House size vs price
plt.subplot(2, 2, 1)
plt.scatter(data['size'], data['price'], alpha=0.5, color='blue')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('House Size vs Price')

# Plot 2: Bedrooms vs price
plt.subplot(2, 2, 2)
plt.scatter(data['bedrooms'], data['price'], alpha=0.5, color='green')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price ($)')
plt.title('Bedrooms vs Price')

# Plot 3: Age vs price
plt.subplot(2, 2, 3)
plt.scatter(data['age'], data['price'], alpha=0.5, color='red')
plt.xlabel('House Age (years)')
plt.ylabel('Price ($)')
plt.title('House Age vs Price')

# Plot 4: Price distribution
plt.subplot(2, 2, 4)
plt.hist(data['price'], bins=30, alpha=0.7, color='purple')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.title('Price Distribution')

plt.tight_layout()
plt.show()

# Step 3: Prepare data for machine learning
print("\nStep 3: Preparing data for ML...")

# Features (X) and target (y)
X = data[['size', 'bedrooms', 'bathrooms', 'age']]
y = data['price']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Step 4: Feature scaling (important for many ML algorithms)
print("\nStep 4: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled to have mean=0 and std=1")

# Step 5: Train the machine learning model
print("\nStep 5: Training the model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")
print(f"Model coefficients: {model.coef_}")
print(f"Model intercept: {model.intercept_}")

# Step 6: Make predictions
print("\nStep 6: Making predictions...")
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Step 7: Evaluate the model
print("\nStep 7: Evaluating model performance...")

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: ${train_mse:,.2f}")
print(f"Testing MSE: ${test_mse:,.2f}")
print(f"Training R²: {train_r2:.3f}")
print(f"Testing R²: {test_r2:.3f}")

# Step 8: Visualize predictions
print("\nStep 8: Visualizing predictions...")
plt.figure(figsize=(10, 6))

# Plot actual vs predicted prices
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted Prices')

# Plot residuals (prediction errors)
plt.subplot(1, 2, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price ($)')
plt.ylabel('Residuals ($)')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()

# Step 9: Make predictions for new houses
print("\nStep 9: Predicting prices for new houses...")

# Example new houses
new_houses = pd.DataFrame({
    'size': [1800, 2500, 3000],
    'bedrooms': [3, 4, 5],
    'bathrooms': [2, 3, 4],
    'age': [5, 10, 2]
})

print("New houses to predict:")
print(new_houses)

# Scale the new data using the same scaler
new_houses_scaled = scaler.transform(new_houses)

# Make predictions
new_predictions = model.predict(new_houses_scaled)

print("\nPredicted prices:")
for i, price in enumerate(new_predictions):
    print(f"House {i+1}: ${price:,.2f}")

# Step 10: Understanding what we learned
print("\n" + "="*50)
print("WHAT WE LEARNED:")
print("="*50)
print("1. DATA PREPARATION:")
print("   - Created a dataset with features (size, bedrooms, bathrooms, age)")
print("   - Split data into training (80%) and testing (20%) sets")
print("   - Scaled features to improve model performance")
print()
print("2. MODEL TRAINING:")
print("   - Used Linear Regression to find relationships between features and price")
print("   - The model learned coefficients for each feature")
print()
print("3. EVALUATION:")
print(f"   - R² score of {test_r2:.3f} means our model explains {test_r2*100:.1f}% of price variation")
print(f"   - MSE of ${test_mse:,.2f} shows average prediction error")
print()
print("4. KEY ML CONCEPTS COVERED:")
print("   - Supervised learning (learning from labeled examples)")
print("   - Train/test split (avoiding overfitting)")
print("   - Feature scaling (preprocessing)")
print("   - Model evaluation (metrics)")
print("   - Prediction (applying the model to new data)")

# Bonus: Feature importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE:")
print("="*50)
feature_names = ['size', 'bedrooms', 'bathrooms', 'age']
coefficients = model.coef_
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.2f} (${coef:.2f} per unit increase)")