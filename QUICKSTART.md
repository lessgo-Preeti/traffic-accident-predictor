# 🚀 Quick Start Guide

## ⚡ Get Started in 5 Minutes!

Follow these simple steps to run the complete project:

---

## 📋 Prerequisites

- Python 3.9 or higher installed
- pip package manager
- VS Code (optional but recommended)

---

## 🔧 Step 1: Install Dependencies

Open PowerShell in project directory:

```powershell
# Navigate to project
cd e:\projects\traffic-accident-predictor

# Install required packages
pip install -r requirements.txt
```

**Installation time**: ~2-3 minutes

---

## 📊 Step 2: Generate Sample Data (OPTION A - Fastest!)

If you don't have real dataset yet:

```powershell
python src\generate_sample_data.py
```

**Output**: Creates `data/sample/india_accidents_sample.csv` with 5000 records

**Time**: 10 seconds ⚡

---

## 📥 Step 2: Download Real Data (OPTION B - Better Accuracy)

1. Visit: https://www.kaggle.com/datasets/tsiaras/india-road-accidents
2. Download CSV file
3. Place in `data\raw\` folder
4. Rename to `india_road_accidents.csv`

---

## 🧹 Step 3: Preprocess Data

```powershell
python src\data_preprocessing.py
```

**What it does**:
- Loads raw/sample data
- Cleans missing values
- Creates new features
- Saves processed data to `data/processed/`

**Time**: 30 seconds - 2 minutes (depending on data size)

---

## 🤖 Step 4: Train ML Models

```powershell
python src\model_training.py
```

**What it does**:
- Trains Random Forest classifier
- Trains XGBoost classifier (if installed)
- Trains Decision Tree (baseline)
- Evaluates all models
- Saves best model to `models/`
- Creates confusion matrix visualizations

**Time**: 2-5 minutes

**Expected Output**:
```
🌲 TRAINING RANDOM FOREST CLASSIFIER
✅ Random Forest trained in 3.45 seconds
   Accuracy: 88.25%

⚡ TRAINING XGBOOST CLASSIFIER
✅ XGBoost trained in 5.12 seconds
   Accuracy: 89.80%

🏆 Best Model: XGBoost (Accuracy: 89.80%)
```

---

## 🎨 Step 5: Run Dashboard

```powershell
streamlit run dashboard\app.py
```

**What it does**:
- Opens interactive web dashboard in browser
- URL: http://localhost:8501

**Features**:
- 🏠 Home - Project overview
- 🔮 Predict - Enter accident details and get severity prediction
- 📊 Analytics - Explore data visualizations
- ℹ️ About - Project information

**Time**: Opens in 5 seconds

---

## 📓 Step 6: Explore Jupyter Notebook (Optional)

```powershell
jupyter notebook
```

Then open: `notebooks/01_EDA.ipynb`

**Contains**:
- Detailed data analysis
- Beautiful visualizations
- Statistical insights
- Feature correlations

---

## ✅ Complete Workflow Summary

```powershell
# All commands in sequence
cd e:\projects\traffic-accident-predictor

pip install -r requirements.txt

python src\generate_sample_data.py

python src\data_preprocessing.py

python src\model_training.py

streamlit run dashboard\app.py
```

**Total time**: ~10 minutes from zero to working dashboard! 🎉

---

## 🎯 What You'll Have After This

✅ Working ML model (85-90% accuracy)  
✅ Interactive web dashboard  
✅ Data visualizations  
✅ Model comparison reports  
✅ Complete project for portfolio  

---

## 🐛 Troubleshooting

### Problem: ModuleNotFoundError
**Solution**: 
```powershell
pip install -r requirements.txt
```

### Problem: Data file not found
**Solution**: 
```powershell
python src\generate_sample_data.py
```

### Problem: XGBoost not working
**Solution**: 
```powershell
pip install xgboost
```
Or skip it - Random Forest works great too!

### Problem: Streamlit won't start
**Solution**: 
```powershell
pip install streamlit --upgrade
streamlit run dashboard\app.py
```

---

## 📸 Expected Screenshots

### Dashboard Home
![Dashboard](assets/dashboard_home.png)

### Prediction Interface
![Prediction](assets/prediction_page.png)

### Analytics
![Analytics](assets/analytics_page.png)

---

## 🎓 Learning Path

**Beginner**: Run all steps → Understand what each does  
**Intermediate**: Modify parameters → Experiment with features  
**Advanced**: Add new models → Improve accuracy → Deploy online  

---

## 🚀 Next Level: Deploy Online (FREE!)

### Deploy to Streamlit Cloud:

1. Push code to GitHub
2. Visit: https://streamlit.io/cloud
3. Connect your repo
4. Click "Deploy"
5. Get public URL to share!

**Time**: 5 minutes  
**Cost**: FREE!  

---

## 📌 Quick Commands Reference

| Task | Command |
|------|---------|
| Install packages | `pip install -r requirements.txt` |
| Generate data | `python src\generate_sample_data.py` |
| Preprocess | `python src\data_preprocessing.py` |
| Train models | `python src\model_training.py` |
| Run dashboard | `streamlit run dashboard\app.py` |
| Open Jupyter | `jupyter notebook` |
| Check Python | `python --version` |
| List files | `ls` |

---

## 💡 Pro Tips

1. **Use sample data first** - Get everything working, then add real data
2. **Check each step** - Don't skip error messages
3. **Take screenshots** - For your portfolio
4. **Experiment** - Change parameters and see what happens
5. **Document changes** - Keep notes for your presentation

---

## 🎉 You're Ready!

Start with:
```powershell
python src\generate_sample_data.py
```

Then follow steps 3-5! 🚀

---

## 📧 Need Help?

Check these resources:
- README.md - Detailed documentation
- DATASET_SETUP.md - Data download guide
- Jupyter notebooks - Code examples
- GitHub Issues - Report problems

**Happy Coding!** 🎯
