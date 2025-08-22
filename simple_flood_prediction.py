#!/usr/bin/env python3
"""
Simple Geospatial Flood Prediction Analysis Script (Standard Library Only)

This Python script performs basic geospatial analysis for flood prediction
using only Python standard library modules.

Usage:
    python simple_flood_prediction.py

Author: Generated for geospatial flood prediction analysis
"""

import json
import csv
import math
import random
from typing import List, Dict, Tuple, Optional

class FloodPredictor:
    """Simple flood prediction model using basic algorithms."""
    
    def __init__(self):
        self.weights = {
            'elevation': -0.3,      # Lower elevation = higher risk
            'rainfall': 0.4,        # More rainfall = higher risk
            'distance_to_water': -0.2,  # Closer to water = higher risk
            'soil_clay': 0.1        # Clay soil = slightly higher risk
        }
    
    def predict_risk(self, elevation: float, rainfall: float, 
                    distance_to_water: float, soil_type: str) -> Tuple[float, str]:
        """
        Predict flood risk based on input parameters.
        
        Args:
            elevation: Elevation in meters
            rainfall: Rainfall in mm
            distance_to_water: Distance to nearest water body in km
            soil_type: Type of soil ('clay', 'sandy', 'loam')
        
        Returns:
            Tuple of (risk_score, risk_level)
        """
        # Normalize inputs
        norm_elevation = (elevation - 10) / 10  # Assume 10m baseline
        norm_rainfall = (rainfall - 40) / 20   # Assume 40mm baseline
        norm_distance = (distance_to_water - 1) / 1  # Assume 1km baseline
        soil_clay = 1 if soil_type == 'clay' else 0
        
        # Calculate risk score
        risk = (
            self.weights['elevation'] * norm_elevation +
            self.weights['rainfall'] * norm_rainfall +
            self.weights['distance_to_water'] * norm_distance +
            self.weights['soil_clay'] * soil_clay +
            0.5  # baseline risk
        )
        
        # Ensure risk is between 0 and 1
        risk = max(0, min(1, risk))
        
        # Determine risk level
        if risk > 0.7:
            level = 'High'
        elif risk > 0.4:
            level = 'Medium'
        else:
            level = 'Low'
        
        return risk, level

def create_sample_data() -> List[Dict]:
    """Create sample data for analysis."""
    return [
        {
            'location': 'Location A',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'elevation': 10.5,
            'rainfall_mm': 45.2,
            'soil_type': 'clay',
            'distance_to_water': 0.5,
            'actual_flood_risk': 0.8
        },
        {
            'location': 'Location B',
            'latitude': 40.7589,
            'longitude': -73.9851,
            'elevation': 15.2,
            'rainfall_mm': 38.7,
            'soil_type': 'sandy',
            'distance_to_water': 1.2,
            'actual_flood_risk': 0.4
        },
        {
            'location': 'Location C',
            'latitude': 40.7614,
            'longitude': -73.9776,
            'elevation': 8.7,
            'rainfall_mm': 52.1,
            'soil_type': 'loam',
            'distance_to_water': 0.3,
            'actual_flood_risk': 0.9
        },
        {
            'location': 'Location D',
            'latitude': 40.7505,
            'longitude': -73.9934,
            'elevation': 12.3,
            'rainfall_mm': 41.3,
            'soil_type': 'clay',
            'distance_to_water': 0.8,
            'actual_flood_risk': 0.6
        },
        {
            'location': 'Location E',
            'latitude': 40.7482,
            'longitude': -73.9857,
            'elevation': 9.8,
            'rainfall_mm': 48.9,
            'soil_type': 'sandy',
            'distance_to_water': 0.6,
            'actual_flood_risk': 0.7
        }
    ]

def analyze_data(data: List[Dict]) -> Dict:
    """Perform basic statistical analysis on the data."""
    flood_risks = [item['actual_flood_risk'] for item in data]
    elevations = [item['elevation'] for item in data]
    rainfalls = [item['rainfall_mm'] for item in data]
    
    stats = {
        'total_locations': len(data),
        'avg_flood_risk': sum(flood_risks) / len(flood_risks),
        'max_flood_risk': max(flood_risks),
        'min_flood_risk': min(flood_risks),
        'avg_elevation': sum(elevations) / len(elevations),
        'avg_rainfall': sum(rainfalls) / len(rainfalls),
        'high_risk_count': sum(1 for risk in flood_risks if risk > 0.7),
        'medium_risk_count': sum(1 for risk in flood_risks if 0.4 < risk <= 0.7),
        'low_risk_count': sum(1 for risk in flood_risks if risk <= 0.4)
    }
    
    return stats

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula."""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

def save_to_csv(data: List[Dict], filename: str = 'flood_data.csv'):
    """Save data to CSV file."""
    if not data:
        return
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Data saved to {filename}")

def save_to_json(data: List[Dict], filename: str = 'flood_data.json'):
    """Save data to JSON file."""
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=2)
    
    print(f"Data saved to {filename}")

def main():
    """Main execution function."""
    print("Simple Geospatial Flood Prediction Analysis")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    print(f"Created sample data with {len(data)} locations")
    
    # Analyze data
    stats = analyze_data(data)
    
    print(f"\n=== Data Analysis Results ===")
    print(f"Total locations: {stats['total_locations']}")
    print(f"Average flood risk: {stats['avg_flood_risk']:.3f}")
    print(f"Risk range: {stats['min_flood_risk']:.3f} - {stats['max_flood_risk']:.3f}")
    print(f"Average elevation: {stats['avg_elevation']:.1f}m")
    print(f"Average rainfall: {stats['avg_rainfall']:.1f}mm")
    print(f"Risk distribution:")
    print(f"  High risk (>0.7): {stats['high_risk_count']} locations")
    print(f"  Medium risk (0.4-0.7): {stats['medium_risk_count']} locations")
    print(f"  Low risk (<=0.4): {stats['low_risk_count']} locations")
    
    # Initialize predictor
    predictor = FloodPredictor()
    
    print(f"\n=== Model Predictions vs Actual ===")
    total_error = 0
    
    for item in data:
        predicted_risk, predicted_level = predictor.predict_risk(
            item['elevation'],
            item['rainfall_mm'],
            item['distance_to_water'],
            item['soil_type']
        )
        
        actual_risk = item['actual_flood_risk']
        error = abs(predicted_risk - actual_risk)
        total_error += error
        
        print(f"{item['location']}: Predicted={predicted_risk:.3f} ({predicted_level}), "
              f"Actual={actual_risk:.3f}, Error={error:.3f}")
    
    avg_error = total_error / len(data)
    print(f"\nAverage prediction error: {avg_error:.3f}")
    
    # Example prediction for new location
    print(f"\n=== Example New Prediction ===")
    new_risk, new_level = predictor.predict_risk(
        elevation=8.0,
        rainfall=55.0,
        distance_to_water=0.4,
        soil_type='clay'
    )
    print(f"New location prediction: {new_risk:.3f} ({new_level} risk)")
    print("Parameters: 8.0m elevation, 55.0mm rainfall, 0.4km from water, clay soil")
    
    # Calculate distances between locations
    print(f"\n=== Distance Analysis ===")
    for i, loc1 in enumerate(data[:3]):  # Just first 3 for brevity
        for j, loc2 in enumerate(data[i+1:4], i+1):
            dist = calculate_distance(
                loc1['latitude'], loc1['longitude'],
                loc2['latitude'], loc2['longitude']
            )
            print(f"Distance {loc1['location']} to {loc2['location']}: {dist:.2f} km")
    
    # Save results
    enhanced_data = []
    for item in data:
        enhanced_item = item.copy()
        pred_risk, pred_level = predictor.predict_risk(
            item['elevation'], item['rainfall_mm'],
            item['distance_to_water'], item['soil_type']
        )
        enhanced_item['predicted_risk'] = round(pred_risk, 3)
        enhanced_item['predicted_level'] = pred_level
        enhanced_data.append(enhanced_item)
    
    save_to_csv(enhanced_data)
    save_to_json(enhanced_data)
    
    print(f"\n" + "=" * 50)
    print("Analysis complete!")
    print("Files generated:")
    print("- flood_data.csv: Data in CSV format")
    print("- flood_data.json: Data in JSON format")

if __name__ == "__main__":
    main()