#!/usr/bin/env python3
"""
Geospatial Flood Prediction Analysis Script

This Python script performs geospatial analysis for flood prediction.
It can be run as a standalone script or imported as a module.

Usage:
    python flood_prediction.py

Author: Generated for geospatial flood prediction analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample data for flood prediction analysis."""
    sample_data = {
        'latitude': [40.7128, 40.7589, 40.7614, 40.7505, 40.7482],
        'longitude': [-74.0060, -73.9851, -73.9776, -73.9934, -73.9857],
        'elevation': [10.5, 15.2, 8.7, 12.3, 9.8],
        'rainfall_mm': [45.2, 38.7, 52.1, 41.3, 48.9],
        'soil_type': ['clay', 'sandy', 'loam', 'clay', 'sandy'],
        'distance_to_water': [0.5, 1.2, 0.3, 0.8, 0.6],
        'flood_risk': [0.8, 0.4, 0.9, 0.6, 0.7]
    }
    return pd.DataFrame(sample_data)

def analyze_flood_data(df):
    """Perform basic analysis on flood data."""
    print("=== Flood Prediction Data Analysis ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Average flood risk: {df['flood_risk'].mean():.3f}")
    print(f"High risk locations (>0.7): {(df['flood_risk'] > 0.7).sum()}")
    print(f"Low risk locations (<0.5): {(df['flood_risk'] < 0.5).sum()}")
    
    print("\nCorrelation with flood risk:")
    correlations = df.select_dtypes(include=[np.number]).corr()['flood_risk'].sort_values(ascending=False)
    for feature, corr in correlations.items():
        if feature != 'flood_risk':
            print(f"  {feature}: {corr:.3f}")

def create_simple_model(df):
    """Create a simple flood prediction model."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Prepare data
        df_encoded = pd.get_dummies(df, columns=['soil_type'], prefix='soil')
        feature_columns = [col for col in df_encoded.columns if col != 'flood_risk']
        X = df_encoded[feature_columns]
        y = df_encoded['flood_risk']
        
        # Generate more synthetic data for training
        np.random.seed(42)
        n_samples = 100
        
        synthetic_X = pd.DataFrame({
            'latitude': np.random.uniform(40.7, 40.8, n_samples),
            'longitude': np.random.uniform(-74.1, -73.9, n_samples),
            'elevation': np.random.uniform(5, 20, n_samples),
            'rainfall_mm': np.random.uniform(30, 60, n_samples),
            'distance_to_water': np.random.uniform(0.1, 2.0, n_samples),
            'soil_clay': np.random.choice([0, 1], n_samples),
            'soil_loam': np.random.choice([0, 1], n_samples),
            'soil_sandy': np.random.choice([0, 1], n_samples)
        })
        
        # Calculate synthetic flood risk
        synthetic_y = (
            0.4 * (synthetic_X['rainfall_mm'] - 30) / 30 +
            0.3 * (20 - synthetic_X['elevation']) / 15 +
            0.2 * (2.0 - synthetic_X['distance_to_water']) / 2.0 +
            0.1 * synthetic_X['soil_clay'] +
            np.random.normal(0, 0.1, n_samples)
        )
        synthetic_y = np.clip(synthetic_y, 0, 1)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            synthetic_X, synthetic_y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n=== Machine Learning Model Results ===")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        for _, row in importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return model, X_train.columns
        
    except ImportError:
        print("\nSkikit-learn not available. Skipping machine learning analysis.")
        return None, None

def predict_flood_risk(model, feature_columns, latitude, longitude, elevation, 
                      rainfall_mm, distance_to_water, soil_type):
    """Predict flood risk for a specific location."""
    if model is None:
        print("Model not available for prediction.")
        return None
    
    # Create input data
    input_data = {
        'latitude': latitude,
        'longitude': longitude,
        'elevation': elevation,
        'rainfall_mm': rainfall_mm,
        'distance_to_water': distance_to_water,
        'soil_clay': 1 if soil_type == 'clay' else 0,
        'soil_loam': 1 if soil_type == 'loam' else 0,
        'soil_sandy': 1 if soil_type == 'sandy' else 0
    }
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    prediction = max(0, min(1, prediction))  # Ensure 0-1 range
    
    risk_level = 'High' if prediction > 0.7 else 'Medium' if prediction > 0.4 else 'Low'
    
    return prediction, risk_level

def main():
    """Main execution function."""
    print("Geospatial Flood Prediction Analysis")
    print("=" * 40)
    
    # Create and analyze sample data
    df = create_sample_data()
    print("Sample data created:")
    print(df.head())
    print()
    
    # Perform analysis
    analyze_flood_data(df)
    
    # Create machine learning model
    model, feature_columns = create_simple_model(df)
    
    # Example prediction
    if model is not None:
        print(f"\n=== Example Prediction ===")
        risk, level = predict_flood_risk(
            model, feature_columns,
            latitude=40.7128, longitude=-74.0060, elevation=8.0,
            rainfall_mm=50.0, distance_to_water=0.3, soil_type='clay'
        )
        print(f"Predicted flood risk: {risk:.3f} ({level} risk)")
    
    print("\n" + "=" * 40)
    print("Analysis complete!")

if __name__ == "__main__":
    main()