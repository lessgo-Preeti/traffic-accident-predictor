# 📥 Dataset Download & Setup Guide

## 🎯 Kaggle Dataset Download Instructions

### **Step 1: Create Kaggle Account** (if not already)
1. Visit: https://www.kaggle.com/
2. Click "Register" 
3. Sign up with Google/Email (takes 1 minute)

---

### **Step 2: Download India Road Accidents Dataset**

#### **Option A: Direct Download (Easiest)**
1. Go to: https://www.kaggle.com/datasets/tsiaras/india-road-accidents
2. Click the **"Download"** button (top right)
3. Save the ZIP file to your Downloads folder
4. Extract the ZIP file
5. You'll get a CSV file (usually named `india_road_accidents.csv` or similar)

#### **Option B: Using Kaggle API** (Advanced)
```powershell
# Install Kaggle
pip install kaggle

# Download dataset
kaggle datasets download -d tsiaras/india-road-accidents

# Extract
Expand-Archive -Path india-road-accidents.zip -DestinationPath data/raw/
```

---

### **Step 3: Move File to Project**

**After extracting the CSV file:**

1. **Find your downloaded CSV file**
   - Usually in: `C:\Users\YourName\Downloads\`
   - File name: `india_road_accidents.csv` or `road_accidents_india.csv`

2. **Copy to project folder**
   ```
   Copy the CSV file to:
   e:\projects\traffic-accident-predictor\data\raw\
   ```

3. **Rename if needed** (optional)
   - Recommended name: `india_road_accidents.csv`

---

### **Step 4: Verify File Location**

Open PowerShell in your project directory and run:

```powershell
# Navigate to project
cd e:\projects\traffic-accident-predictor

# Check if file exists
ls data\raw\

# Should see: india_road_accidents.csv
```

---

## 📁 Final Directory Structure Should Look Like:

```
e:\projects\traffic-accident-predictor\
├── data\
│   ├── raw\
│   │   └── india_road_accidents.csv  ← Your dataset here!
│   ├── processed\
│   └── sample\
├── src\
├── models\
├── notebooks\
└── ...
```

---

## 🚨 Troubleshooting

### **Problem 1: File too large / Download fails**
**Solution**: Try alternative datasets:
- https://www.kaggle.com/datasets/saisampathkumar/road-accident-deaths-in-india
- https://data.gov.in/ (search "road accidents")

### **Problem 2: Don't have Kaggle account**
**Solution**: 
- Use our sample data generator: `python src/generate_sample_data.py`
- Or download from data.gov.in (no account needed)

### **Problem 3: CSV file has different name**
**Solution**: Either:
- Rename it to `india_road_accidents.csv`
- Or update the file path in `src/data_preprocessing.py`

---

## ✅ Alternative: Use Sample Data (No Download Needed!)

If you want to start immediately without downloading:

```powershell
# Generate 5000 sample records
python src\generate_sample_data.py
```

This creates: `data\sample\india_accidents_sample.csv`

---

## 📌 Next Steps After Download

Once your CSV is in `data\raw\`:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run preprocessing
python src\data_preprocessing.py

# 3. Train model
python src\model_training.py

# 4. Run dashboard
streamlit run dashboard\app.py
```

---

## 💡 Quick Start (If You Can't Download Dataset Now)

**Use sample data to start learning immediately:**

```powershell
# Generate sample data
python src\generate_sample_data.py

# This will create data in data\sample\ folder
# You can use this to test the entire pipeline!
```

**Later**, when you download real data, just copy it to `data\raw\` and re-run everything!

---

## 🆘 Need Help?

If file download/placement is confusing, just run:
```powershell
python src\generate_sample_data.py
```

And start working immediately with sample data! 🚀
