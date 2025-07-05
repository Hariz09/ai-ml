# Course 2: Advanced House Price Prediction - Multiple Models & Feature Engineering
# Building on Course 1: Compare different algorithms and improve features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

print("COURSE 2: ADVANCED HOUSE PRICE PREDICTION")
print("="*60)
print("Learning objectives:")
print("1. Feature engineering and polynomial features")
print("2. Compare multiple ML algorithms")
print("3. Cross-validation for robust evaluation")
print("4. Regularization techniques (Ridge, Lasso)")
print("5. Tree-based models (Decision Trees, Random Forest)")
print("="*60)

# Step 1: Create enhanced dataset with more realistic features
print("\nStep 1: Creating enhanced dataset...")
np.random.seed(42)

n_samples = 2000
house_sizes = np.random.normal(2000, 600, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 5, n_samples)
age = np.random.randint(0, 50, n_samples)
garage_size = np.random.randint(0, 4, n_samples)  # New feature
lot_size = np.random.normal(8000, 2000, n_samples)  # New feature
neighborhood_quality = np.random.randint(1, 11, n_samples)  # New feature (1-10 scale)

# More complex price formula with interactions
prices = (house_sizes * 120 + 
          bedrooms * 8000 + 
          bathrooms * 12000 + 
          garage_size * 15000 +
          lot_size * 2 +
          neighborhood_quality * 10000 -
          age * 1200 +
          house_sizes * neighborhood_quality * 0.05 +  # Interaction term
          np.random.normal(0, 25000, n_samples))

data = pd.DataFrame({
    'size': house_sizes,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'garage_size': garage_size,
    'lot_size': lot_size,
    'neighborhood_quality': neighborhood_quality,
    'price': prices
})

print(f"Enhanced dataset created with {len(data)} houses and {len(data.columns)-1} features")
print("\nNew dataset overview:")
print(data.head())
print("\nDataset statistics:")
print(data.describe())

# Step 2: Advanced data visualization
print("\nStep 2: Advanced data visualization...")
plt.figure(figsize=(15, 10))

# Correlation heatmap
plt.subplot(2, 3, 1)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')

# Distribution plots
plt.subplot(2, 3, 2)
plt.hist(data['price'], bins=50, alpha=0.7, color='skyblue')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.title('Price Distribution')

# Box plot by neighborhood quality
plt.subplot(2, 3, 3)
data.boxplot(column='price', by='neighborhood_quality', ax=plt.gca())
plt.title('Price by Neighborhood Quality')
plt.xlabel('Neighborhood Quality (1-10)')

# Scatter plot with color coding
plt.subplot(2, 3, 4)
scatter = plt.scatter(data['size'], data['price'], 
                     c=data['neighborhood_quality'], 
                     cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Neighborhood Quality')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Size vs Price (colored by quality)')

# Age vs price with garage size
plt.subplot(2, 3, 5)
for garage in range(4):
    subset = data[data['garage_size'] == garage]
    plt.scatter(subset['age'], subset['price'], 
               alpha=0.6, label=f'{garage} car garage')
plt.xlabel('Age (years)')
plt.ylabel('Price ($)')
plt.title('Age vs Price by Garage Size')
plt.legend()

# Feature importance preview
plt.subplot(2, 3, 6)
feature_cols = ['size', 'bedrooms', 'bathrooms', 'age', 'garage_size', 'lot_size', 'neighborhood_quality']
correlations = [abs(data[col].corr(data['price'])) for col in feature_cols]
plt.barh(feature_cols, correlations)
plt.xlabel('Absolute Correlation with Price')
plt.title('Feature Importance (Correlation)')

plt.tight_layout()
plt.show()

# Step 3: Feature Engineering
print("\nStep 3: Feature Engineering...")

# Create new features
data['price_per_sqft'] = data['price'] / data['size']
data['total_rooms'] = data['bedrooms'] + data['bathrooms']
data['size_age_ratio'] = data['size'] / (data['age'] + 1)  # +1 to avoid division by zero
data['luxury_score'] = (data['neighborhood_quality'] * data['garage_size'] * 
                       data['bathrooms']) / (data['age'] + 1)

# Prepare features and target
feature_columns = ['size', 'bedrooms', 'bathrooms', 'age', 'garage_size', 
                  'lot_size', 'neighborhood_quality', 'total_rooms', 
                  'size_age_ratio', 'luxury_score']
X = data[feature_columns]
y = data['price']

print(f"Features after engineering: {X.shape[1]} features")
print("New features created:")
print("- total_rooms: bedrooms + bathrooms")
print("- size_age_ratio: size / (age + 1)")
print("- luxury_score: complex interaction term")

# Step 4: Data splitting and preprocessing
print("\nStep 4: Data preparation...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Step 5: Multiple Model Comparison
print("\nStep 5: Training and comparing multiple models...")

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
}

# Train and evaluate each model
results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                               scoring='neg_mean_squared_error')
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    results[name] = {
        'CV_Score': -cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Test_MSE': test_mse,
        'Test_MAE': test_mae
    }
    
    predictions[name] = y_test_pred
    
    print(f"  Cross-validation MSE: ${-cv_scores.mean():,.2f} (±{cv_scores.std():,.2f})")
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  Test MSE: ${test_mse:,.2f}")

# Step 6: Results comparison
print("\nStep 6: Model Comparison Results")
print("="*60)
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('Test_R2', ascending=False)
print(results_df.round(3))

# Step 7: Visualization of results
print("\nStep 7: Visualizing model performance...")
plt.figure(figsize=(15, 10))

# R² comparison
plt.subplot(2, 3, 1)
model_names = list(results.keys())
test_r2_scores = [results[name]['Test_R2'] for name in model_names]
plt.barh(model_names, test_r2_scores)
plt.xlabel('R² Score')
plt.title('Model Performance Comparison (R²)')

# MSE comparison
plt.subplot(2, 3, 2)
test_mse_scores = [results[name]['Test_MSE'] for name in model_names]
plt.barh(model_names, test_mse_scores)
plt.xlabel('MSE')
plt.title('Model Performance Comparison (MSE)')

# Actual vs Predicted for best model
best_model = results_df.index[0]
plt.subplot(2, 3, 3)
plt.scatter(y_test, predictions[best_model], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Actual vs Predicted ({best_model})')

# Residuals for best model
plt.subplot(2, 3, 4)
residuals = y_test - predictions[best_model]
plt.scatter(predictions[best_model], residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price ($)')
plt.ylabel('Residuals ($)')
plt.title(f'Residual Plot ({best_model})')

# Cross-validation scores
plt.subplot(2, 3, 5)
cv_means = [results[name]['CV_Score'] for name in model_names]
cv_stds = [results[name]['CV_Std'] for name in model_names]
plt.barh(model_names, cv_means, xerr=cv_stds)
plt.xlabel('Cross-Validation MSE')
plt.title('Cross-Validation Performance')

# Feature importance for Random Forest
plt.subplot(2, 3, 6)
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_importance = rf_model.feature_importances_
    plt.barh(feature_columns, feature_importance)
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')

plt.tight_layout()
plt.show()

# Step 8: Polynomial Features Example
print("\nStep 8: Exploring Polynomial Features...")
print("Testing polynomial features with Linear Regression...")

# Create polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

print(f"Original features: {X_train_scaled.shape[1]}")
print(f"Polynomial features: {X_train_poly.shape[1]}")

# Train model with polynomial features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Evaluate
poly_pred = poly_model.predict(X_test_poly)
poly_r2 = r2_score(y_test, poly_pred)
poly_mse = mean_squared_error(y_test, poly_pred)

print(f"Polynomial Model R²: {poly_r2:.3f}")
print(f"Polynomial Model MSE: ${poly_mse:,.2f}")

# Step 9: Regularization Comparison
print("\nStep 9: Regularization techniques comparison...")
alphas = [0.1, 1.0, 10.0, 100.0]
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge_cv = cross_val_score(ridge, X_train_scaled, y_train, cv=5)
    ridge_scores.append(ridge_cv.mean())
    
    # Lasso
    lasso = Lasso(alpha=alpha)
    lasso_cv = cross_val_score(lasso, X_train_scaled, y_train, cv=5)
    lasso_scores.append(lasso_cv.mean())

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_scores, 'o-', label='Ridge', linewidth=2)
plt.plot(alphas, lasso_scores, 's-', label='Lasso', linewidth=2)
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Cross-Validation Score')
plt.title('Regularization Parameter Tuning')
plt.legend()
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.show()

# Step 10: Key Learnings Summary
print("\n" + "="*60)
print("COURSE 2 KEY LEARNINGS:")
print("="*60)
print("1. FEATURE ENGINEERING:")
print("   - Created interaction terms and derived features")
print("   - Feature importance helps identify most valuable predictors")
print("   - More features can improve performance but may cause overfitting")
print()
print("2. MULTIPLE ALGORITHMS:")
print("   - Linear models: Simple, interpretable, good baseline")
print("   - Tree models: Handle non-linearity, feature interactions")
print("   - Ensemble methods: Often achieve best performance")
print()
print("3. MODEL EVALUATION:")
print("   - Cross-validation provides robust performance estimates")
print("   - Multiple metrics give different perspectives on performance")
print("   - Compare training vs test performance to detect overfitting")
print()
print("4. REGULARIZATION:")
print("   - Ridge: Reduces overfitting, keeps all features")
print("   - Lasso: Can eliminate features, automatic feature selection")
print("   - Alpha parameter controls regularization strength")
print()
print("5. ADVANCED CONCEPTS:")
print("   - Polynomial features capture non-linear relationships")
print("   - Feature scaling is crucial for distance-based algorithms")
print("   - Model complexity vs. generalization trade-off")
print()
print("BEST PERFORMING MODEL:")
print(f"Model: {best_model}")
print(f"Test R²: {results[best_model]['Test_R2']:.3f}")
print(f"Test MSE: ${results[best_model]['Test_MSE']:,.2f}")
print()
print("NEXT STEPS FOR COURSE 3:")
print("- Classification problems")
print("- Advanced ensemble methods")
print("- Model hyperparameter tuning")
print("- Real-world data challenges")