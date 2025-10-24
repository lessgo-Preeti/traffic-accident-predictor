"""
Indian Road Accident Sample Data Generator
-------------------------------------------
Generate realistic sample data for testing the ML pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Indian states and major cities
STATES = {
    'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Thane'],
    'Karnataka': ['Bangalore', 'Mysore', 'Mangalore', 'Hubli'],
    'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem'],
    'Delhi': ['New Delhi', 'South Delhi', 'North Delhi', 'East Delhi'],
    'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot'],
    'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota'],
    'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Agra', 'Varanasi'],
    'West Bengal': ['Kolkata', 'Howrah', 'Durgapur', 'Siliguri'],
    'Kerala': ['Kochi', 'Thiruvananthapuram', 'Kozhikode', 'Thrissur'],
    'Punjab': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala']
}

WEATHER_CONDITIONS = ['Clear', 'Rainy', 'Foggy', 'Cloudy', 'Heavy Rain']
ROAD_TYPES = ['National Highway', 'State Highway', 'City Road', 'Village Road', 'Expressway']
VEHICLE_TYPES = ['Two Wheeler', 'Car', 'Bus', 'Truck', 'Auto Rickshaw', 'Taxi']
ROAD_CONDITIONS = ['Dry', 'Wet', 'Damaged', 'Under Construction']
ACCIDENT_CAUSES = ['Speeding', 'Drunk Driving', 'No Helmet', 'Signal Jump', 'Wrong Side', 
                   'Pothole', 'Animal Crossing', 'Fog', 'Heavy Rain', 'Driver Fatigue']

def generate_sample_data(n_samples=5000):
    """
    Generate sample Indian road accident data
    
    Parameters:
    -----------
    n_samples : int
        Number of accident records to generate
    
    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    
    print(f"🚦 Generating {n_samples} sample Indian accident records...")
    
    data = []
    
    # Start date: 3 years ago
    start_date = datetime.now() - timedelta(days=3*365)
    
    for i in range(n_samples):
        # Random state and city
        state = random.choice(list(STATES.keys()))
        city = random.choice(STATES[state])
        
        # Random date and time
        random_days = random.randint(0, 3*365)
        accident_date = start_date + timedelta(days=random_days)
        hour = random.randint(0, 23)
        
        # Weather (more likely to be clear)
        weather_weights = [0.5, 0.2, 0.1, 0.15, 0.05]
        weather = np.random.choice(WEATHER_CONDITIONS, p=weather_weights)
        
        # Road type
        road_type = random.choice(ROAD_TYPES)
        
        # Vehicle type (two-wheelers are most common in India)
        vehicle_weights = [0.4, 0.3, 0.1, 0.1, 0.05, 0.05]
        vehicle_type = np.random.choice(VEHICLE_TYPES, p=vehicle_weights)
        
        # Road condition
        road_condition = random.choice(ROAD_CONDITIONS)
        if weather in ['Rainy', 'Heavy Rain']:
            road_condition = 'Wet'
        
        # Accident cause
        cause = random.choice(ACCIDENT_CAUSES)
        
        # Traffic signal (more likely on city roads)
        traffic_signal = 1 if road_type in ['City Road', 'National Highway'] and random.random() > 0.5 else 0
        
        # Speed breaker
        speed_breaker = 1 if road_type in ['City Road', 'Village Road'] and random.random() > 0.7 else 0
        
        # Railway crossing
        railway_crossing = 1 if random.random() > 0.9 else 0
        
        # Zebra crossing
        zebra_crossing = 1 if road_type == 'City Road' and random.random() > 0.6 else 0
        
        # Number of vehicles involved
        vehicles_involved = random.randint(1, 4)
        
        # Weather parameters
        if weather == 'Foggy':
            visibility = random.randint(10, 50)  # Low visibility
            temperature = random.randint(10, 20)
            humidity = random.randint(70, 95)
        elif weather in ['Rainy', 'Heavy Rain']:
            visibility = random.randint(30, 70)
            temperature = random.randint(20, 30)
            humidity = random.randint(80, 100)
        else:
            visibility = random.randint(80, 100)
            temperature = random.randint(15, 40)
            humidity = random.randint(40, 80)
        
        # Casualties (depends on vehicle type and severity)
        # Determine severity first (for logic)
        severity_score = 0
        
        # Factors increasing severity
        if vehicle_type == 'Two Wheeler':
            severity_score += 2
        if weather in ['Foggy', 'Heavy Rain']:
            severity_score += 1
        if hour >= 22 or hour <= 5:  # Night time
            severity_score += 1
        if cause in ['Drunk Driving', 'Speeding', 'Signal Jump']:
            severity_score += 2
        if road_type == 'National Highway':
            severity_score += 1
        if vehicles_involved > 2:
            severity_score += 1
        
        # Map to severity (1-4)
        if severity_score <= 2:
            severity = 1  # Minor
            fatalities = 0
            grievous_injuries = random.randint(0, 1)
            minor_injuries = random.randint(1, 2)
        elif severity_score <= 4:
            severity = 2  # Moderate
            fatalities = random.randint(0, 1)
            grievous_injuries = random.randint(1, 2)
            minor_injuries = random.randint(1, 3)
        elif severity_score <= 6:
            severity = 3  # Severe
            fatalities = random.randint(1, 2)
            grievous_injuries = random.randint(1, 3)
            minor_injuries = random.randint(2, 5)
        else:
            severity = 4  # Fatal
            fatalities = random.randint(2, 5)
            grievous_injuries = random.randint(2, 4)
            minor_injuries = random.randint(3, 8)
        
        # Total casualties
        total_casualties = fatalities + grievous_injuries + minor_injuries
        
        # Create record
        record = {
            'Accident_ID': f'ACC{i+1:06d}',
            'State': state,
            'City': city,
            'Date': accident_date.strftime('%Y-%m-%d'),
            'Time': f'{hour:02d}:{random.randint(0, 59):02d}',
            'Day': accident_date.day,
            'Month': accident_date.month,
            'Year': accident_date.year,
            'Hour': hour,
            'DayOfWeek': accident_date.weekday(),
            'Weekend': 1 if accident_date.weekday() >= 5 else 0,
            'Weather_Condition': weather,
            'Temperature': temperature,
            'Humidity': humidity,
            'Visibility': visibility,
            'Road_Type': road_type,
            'Road_Condition': road_condition,
            'Vehicle_Type': vehicle_type,
            'Vehicles_Involved': vehicles_involved,
            'Traffic_Signal': traffic_signal,
            'Speed_Breaker': speed_breaker,
            'Railway_Crossing': railway_crossing,
            'Zebra_Crossing': zebra_crossing,
            'Accident_Cause': cause,
            'Fatalities': fatalities,
            'Grievous_Injuries': grievous_injuries,
            'Minor_Injuries': minor_injuries,
            'Total_Casualties': total_casualties,
            'Severity': severity,
            'Latitude': round(random.uniform(8.0, 35.0), 6),  # India lat range
            'Longitude': round(random.uniform(68.0, 97.0), 6)  # India long range
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    print("✅ Sample data generated successfully!")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Records: {len(df)}")
    print(f"   States: {df['State'].nunique()}")
    print(f"   Cities: {df['City'].nunique()}")
    print(f"   Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\n   Severity Distribution:")
    print(df['Severity'].value_counts().sort_index())
    
    return df


if __name__ == "__main__":
    import os
    
    print("="*60)
    print("🇮🇳 INDIAN ROAD ACCIDENT SAMPLE DATA GENERATOR")
    print("="*60)
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/sample', exist_ok=True)
    
    # Generate sample data
    df = generate_sample_data(n_samples=5000)
    
    # Save to CSV
    output_file = 'data/sample/india_accidents_sample.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved to: {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    print("\n✅ Sample data ready!")
    print("\n📌 Next steps:")
    print("   1. Run: python src/data_preprocessing.py")
    print("   2. Or download real dataset from Kaggle/data.gov.in")
    
    # Display sample records
    print("\n📋 Sample Records:")
    print(df.head(3))
