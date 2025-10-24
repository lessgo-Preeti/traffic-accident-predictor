"""
Data Preprocessing Module
--------------------------
Functions for cleaning and preprocessing traffic accident data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path, sample_size=None):
    """
    Load accident data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    sample_size : int, optional
        Number of rows to sample (for faster testing)
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    print("📂 Loading data...")
    
    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size)
        print(f"✅ Loaded {sample_size} rows (sample)")
    else:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} rows")
    
    return df


def clean_data(df):
    """
    Clean and preprocess the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("\n🧹 Cleaning data...")
    
    # Create a copy
    df_clean = df.copy()
    
    # 1. Handle missing values
    print("  - Handling missing values...")
    
    # Drop columns with >50% missing values
    missing_threshold = 0.5
    missing_pct = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    df_clean = df_clean.drop(columns=cols_to_drop)
    print(f"    Dropped {len(cols_to_drop)} columns with >50% missing")
    
    # Fill missing values for important columns
    if 'Temperature' in df_clean.columns:
        df_clean['Temperature'].fillna(df_clean['Temperature'].median(), inplace=True)
    
    if 'Humidity' in df_clean.columns:
        df_clean['Humidity'].fillna(df_clean['Humidity'].median(), inplace=True)
    
    if 'Visibility' in df_clean.columns:
        df_clean['Visibility'].fillna(df_clean['Visibility'].median(), inplace=True)
    
    if 'Weather_Condition' in df_clean.columns:
        df_clean['Weather_Condition'].fillna('Clear', inplace=True)
    
    if 'Road_Type' in df_clean.columns:
        df_clean['Road_Type'].fillna('City Road', inplace=True)
    
    # 2. Remove duplicates
    print("  - Removing duplicates...")
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    print(f"    Removed {before - after} duplicate rows")
    
    # 3. Handle outliers in numeric columns
    print("  - Handling outliers...")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col != 'Severity':  # Don't touch target variable
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            df_clean[col] = df_clean[col].clip(lower, upper)
    
    print(f"✅ Cleaned data shape: {df_clean.shape}")
    
    return df_clean


def feature_engineering(df):
    """
    Create new features from existing ones
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    print("\n🔧 Engineering features...")
    
    df_feat = df.copy()
    
    # 1. Time-based features
    if 'Start_Time' in df_feat.columns:
        df_feat['Start_Time'] = pd.to_datetime(df_feat['Start_Time'], errors='coerce')
        
        df_feat['Hour'] = df_feat['Start_Time'].dt.hour
        df_feat['Day'] = df_feat['Start_Time'].dt.day
        df_feat['Month'] = df_feat['Start_Time'].dt.month
        df_feat['Year'] = df_feat['Start_Time'].dt.year
        df_feat['DayOfWeek'] = df_feat['Start_Time'].dt.dayofweek  # 0=Monday, 6=Sunday
        df_feat['Weekend'] = (df_feat['DayOfWeek'] >= 5).astype(int)
        
        # Time of day categories
        df_feat['TimeOfDay'] = pd.cut(df_feat['Hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                       include_lowest=True)
        
        print("  ✓ Created time-based features")
    
    # 2. Weather severity score
    weather_features = []
    if 'Temperature' in df_feat.columns:
        weather_features.append('Temperature')
    if 'Visibility' in df_feat.columns:
        weather_features.append('Visibility')
    if 'Humidity' in df_feat.columns:
        weather_features.append('Humidity')
    
    if weather_features:
        # Normalize and create composite score
        df_feat['Weather_Severity_Score'] = 0
        
        if 'Visibility' in df_feat.columns:
            # Lower visibility = higher risk
            df_feat['Weather_Severity_Score'] += (100 - df_feat['Visibility'].clip(0, 100)) / 100
        
        if 'Humidity' in df_feat.columns:
            df_feat['Weather_Severity_Score'] += df_feat['Humidity'] / 100
        
        print("  ✓ Created weather severity score")
    
    # 3. Road and vehicle features
    # Indian-specific features
    india_road_features = ['Highway', 'National_Highway', 'State_Highway', 
                           'City_Road', 'Village_Road', 'Traffic_Signal',
                           'Zebra_Crossing', 'Speed_Breaker', 'Railway_Crossing']
    
    available_road_features = [f for f in india_road_features if f in df_feat.columns]
    
    if available_road_features:
        df_feat['Road_Features_Count'] = df_feat[available_road_features].sum(axis=1)
        print(f"  ✓ Created road features count ({len(available_road_features)} features)")
    
    # Vehicle type encoding
    if 'Vehicle_Type' in df_feat.columns:
        vehicle_risk = {
            'Two Wheeler': 3,  # High risk
            'Motorcycle': 3,
            'Scooter': 3,
            'Car': 2,          # Medium risk
            'Auto Rickshaw': 2,
            'Taxi': 2,
            'Bus': 1,          # Lower risk (but more casualties)
            'Truck': 1,
            'Other': 2
        }
        df_feat['Vehicle_Risk_Score'] = df_feat['Vehicle_Type'].map(vehicle_risk).fillna(2)
        print("  ✓ Created vehicle risk score")
    
    # 4. Distance/Impact category
    if 'Distance' in df_feat.columns or 'Impact_Distance' in df_feat.columns:
        dist_col = 'Distance' if 'Distance' in df_feat.columns else 'Impact_Distance'
        df_feat['Distance_Category'] = pd.cut(df_feat[dist_col],
                                               bins=[0, 50, 100, 200, 1000],
                                               labels=['Very_Short', 'Short', 'Medium', 'Long'])
        print("  ✓ Created distance category")
    
    # 5. Indian holiday/festival indicator (if date available)
    if 'Month' in df_feat.columns and 'Day' in df_feat.columns:
        # Major accident-prone periods in India
        df_feat['Festival_Period'] = 0
        # Diwali period (Oct-Nov), Holi (Mar), New Year
        df_feat.loc[(df_feat['Month'].isin([10, 11, 12, 1, 3])), 'Festival_Period'] = 1
        print("  ✓ Created festival period indicator")
    
    print(f"✅ Feature engineering complete. New shape: {df_feat.shape}")
    
    return df_feat


def select_features(df, target_col='Severity'):
    """
    Select relevant features for modeling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with all features
    target_col : str
        Name of target column
    
    Returns:
    --------
    tuple
        (X, y) - Features and target
    """
    print("\n🎯 Selecting features...")
    
    # Features to exclude
    exclude_cols = [
        target_col,
        'ID', 'Accident_ID', 'FIR_Number', 'Start_Time', 'End_Time', 
        'Accident_Time', 'Date', 'Description',
        'Street', 'Road_Name', 'City', 'District', 'State',
        'Pincode', 'Location', 'Latitude', 'Longitude',
        'Police_Station', 'Hospital_Name',
        # Exclude target-related features (data leakage!)
        'Fatalities', 'Grievous_Injuries', 'Minor_Injuries', 'Total_Casualties',
        'Accident_Cause'  # This is also outcome-related
    ]
    
    # Select numeric and boolean features
    feature_cols = []
    
    for col in df.columns:
        if col not in exclude_cols:
            # Keep numeric columns
            if df[col].dtype in ['int64', 'float64', 'bool']:
                feature_cols.append(col)
            # Keep categorical columns with few unique values
            elif df[col].dtype == 'object' and df[col].nunique() < 20:
                feature_cols.append(col)
    
    # Prepare features
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        print(f"  - Encoding {len(categorical_cols)} categorical features...")
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle any remaining missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    
    print(f"✅ Selected {X.shape[1]} features")
    print(f"   Target variable: {target_col} with {y.nunique()} classes")
    
    return X, y


def save_processed_data(X, y, output_path='data/processed/'):
    """
    Save processed data
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    output_path : str
        Directory to save files
    """
    import os
    os.makedirs(output_path, exist_ok=True)
    
    X.to_csv(f"{output_path}/X_features.csv", index=False)
    y.to_csv(f"{output_path}/y_target.csv", index=False)
    
    print(f"\n💾 Saved processed data to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("🚦 INDIAN TRAFFIC ACCIDENT DATA PREPROCESSING")
    print("="*60)
    
    # Note: Update file path to your dataset location
    # Try multiple possible filenames
    possible_files = [
        "data/raw/india_road_accidents.csv",
        "data/raw/road_accidents_india.csv",
        "data/raw/accidents_data.csv",
        "data/sample/india_accidents_sample.csv",
        "data/sample/sample_accidents.csv"
    ]
    
    file_path = None
    for fp in possible_files:
        import os
        if os.path.exists(fp):
            file_path = fp
            break
    
    if file_path is None:
        print("\n⚠️  Dataset not found!")
        print("   Please download India Road Accidents dataset from:")
        print("   - https://www.kaggle.com/datasets/tsiaras/india-road-accidents")
        print("   - https://data.gov.in/")
        print("   And place in data/raw/ folder")
        print("\n   Or run: python src/generate_sample_data.py")
        exit()
    
    # Load data (use sample for testing)
    df = load_data(file_path, sample_size=50000)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Feature engineering
    df_feat = feature_engineering(df_clean)
    
    # Select features
    X, y = select_features(df_feat)
    
    # Save processed data
    save_processed_data(X, y)
    
    print("\n✅ Preprocessing complete!")
    print(f"   Final dataset: {X.shape[0]} rows × {X.shape[1]} features")
    print("\n📌 Next steps:")
    print("   1. Run: python src/model_training.py")
    print("   2. Or open: notebooks/02_model_training.ipynb")
