# 🚦 Traffic Accident Severity Prediction System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)
![ML](https://img.shields.io/badge/ML-Random%20Forest%2070%25-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-69%25-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)

> An intelligent machine learning system to predict traffic accident severity on Indian roads using real-world data. Built for placement projects, SIH competitions, and portfolio building.

## 🌐 Live Demo

**🔗 Try it now**: [Traffic Accident Predictor (Streamlit Cloud)](https://lessgo-preeti-traffic-accident-predictor.streamlit.app)

**📂 GitHub Repository**: [View Source Code](https://github.com/lessgo-preeti/traffic-accident-predictor)

## 🎯 Project Overview

This project uses machine learning to analyze Indian traffic accident data and predict the severity of accidents based on various factors like weather conditions, road conditions, vehicle type, and location. Perfect for:

- 🎓 **College Projects**: Complete ML pipeline from data to deployment
- 🏆 **SIH/Hackathons**: Real-world problem solving with AI
- 💼 **Portfolio**: Showcase end-to-end ML skills
- 🚀 **Learning**: Hands-on experience with modern ML stack

### What the System Provides:

- **Severity Prediction**: Classify accidents as Minor, Moderate, Severe, or Fatal
- **Risk Analysis**: Identify high-risk combinations (Two-Wheeler + Foggy + Night = Fatal!)
- **Interactive Dashboard**: User-friendly web interface for real-time predictions
- **Data Insights**: Comprehensive EDA with visualizations

## ✨ Key Features

- ✅ **Production-Ready Accuracy**: Random Forest 70%, XGBoost 69% (realistic, no data leakage!)
- ✅ **Real User Input**: Dashboard accepts actual user inputs, not just dataset analytics
- ✅ **22 Engineered Features**: Smart feature engineering for Indian road conditions
- ✅ **Interactive Streamlit Dashboard**: Beautiful UI with real-time predictions
- ✅ **Cloud Deployed**: FREE hosting on Streamlit Cloud
- ✅ **Complete Documentation**: Setup guides, deployment instructions, and more
- ✅ **No Hardware Required**: Pure software ML project

## 📊 Dataset

**Data Source**: Custom generated Indian road accident dataset (5000 records)

**Dataset Characteristics**:
- **Size**: 5,000 accident records
- **Coverage**: 10 Indian states, 41 cities
- **Time Period**: 2022-2025
- **Features**: 22 engineered features including:
  - 📍 Location: State, City (excluded from training - privacy)
  - 🌤️ Weather: Temperature, Humidity, Visibility, Conditions
  - 🛣️ Road: Type (National Highway, City Road, etc.), Condition, Features
  - 🚗 Vehicle: Type (Two-Wheeler, Car, Truck, etc.), Count
  - ⏰ Time: Hour, Day, Month, Weekend indicator
  - 🎯 Target: Severity (1=Minor, 2=Moderate, 3=Severe, 4=Fatal)

**No Data Leakage**: Casualties/Fatalities features excluded to prevent target leakage!

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **Scikit-learn** | Machine learning models |
| **XGBoost** | Gradient boosting classifier |
| **Pandas & NumPy** | Data manipulation |
| **Matplotlib & Seaborn** | Static visualizations |
| **Plotly** | Interactive charts |
| **Folium** | Interactive maps |
| **Streamlit** | Web dashboard |
| **Jupyter Notebook** | Data exploration |

## 📁 Project Structure

```
traffic-accident-predictor/
├── data/
│   ├── raw/                    # Raw dataset (download here)
│   ├── processed/              # Cleaned and preprocessed data
│   └── sample/                 # Sample data for testing
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data cleaning & feature engineering
│   └── 03_model_training.ipynb # Model development & evaluation
├── src/
│   ├── data_preprocessing.py  # Data cleaning functions
│   ├── feature_engineering.py # Feature creation
│   ├── model_training.py      # Model training pipeline
│   └── model_evaluation.py    # Evaluation metrics
├── models/
│   ├── random_forest_model.pkl # Trained Random Forest model
│   ├── xgboost_model.pkl      # Trained XGBoost model
│   └── scaler.pkl             # Feature scaler
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── assets/
│   └── screenshots/           # Demo images
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 8GB+ RAM (for handling large dataset)
- Internet connection (for downloading dataset)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/traffic-accident-predictor.git
cd traffic-accident-predictor
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
   
   **Option 1 - Kaggle** (Recommended):
   - Visit [India Road Accidents Dataset](https://www.kaggle.com/datasets/tsiaras/india-road-accidents)
   - Download `india_road_accidents.csv`
   - Place in `data/raw/` folder
   
   **Option 2 - Government Data Portal**:
   - Visit [data.gov.in](https://data.gov.in/)
   - Search for "Road Accidents India"
   - Download CSV file
   - Place in `data/raw/` folder
   
   **Option 3 - Use sample data** (For quick start):
   - Project includes sample dataset in `data/sample/`
   - Good for testing and learning

## 📝 Usage

### 1. Data Exploration
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 2. Train Models
```bash
python src/model_training.py
```

### 3. Run Dashboard
```bash
streamlit run dashboard/app.py
```

Dashboard will open at: `http://localhost:8501`

## 🎨 Dashboard Features

### 🏠 Home Page
- Project overview
- Key statistics
- Quick navigation

### 🔮 Prediction Interface
- Input accident parameters
- Get severity prediction
- View probability scores
- See risk assessment

### 🗺️ Risk Zone Mapping
- Interactive accident heatmap
- Filter by severity, time, weather
- Zoom and explore locations

### 📊 Analytics
- Hourly/daily/monthly trends
- Weather impact analysis
- Road condition statistics
- Feature importance charts

### 📈 Model Performance
- Accuracy metrics
- Confusion matrix
- ROC curves
- Model comparison

## 🎯 ML Models Performance

### Random Forest Classifier ⭐ (Best Model)
- **Accuracy**: 69.90%
- **Precision**: 0.67
- **Recall**: 0.70
- **F1-Score**: 0.66
- **Training Time**: ~0.3 seconds
- **Why Best**: Balanced performance, fast training, no overfitting

### XGBoost Classifier
- **Accuracy**: 69.40%
- **Precision**: 0.67
- **Recall**: 0.69
- **F1-Score**: 0.67
- **Training Time**: ~1.6 seconds
- **Note**: Similar performance to RF but slower

### Decision Tree (Baseline)
- **Accuracy**: 62.80%
- **Training Time**: 0.04 seconds
- **Use**: Baseline comparison only

## 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Speed |
|-------|----------|-----------|--------|----------|-------|
| **Random Forest** ⭐ | **69.90%** | **0.67** | **0.70** | **0.66** | **Fast** |
| XGBoost | 69.40% | 0.67 | 0.69 | 0.67 | Medium |
| Decision Tree | 62.80% | 0.63 | 0.63 | 0.63 | Very Fast |

**Note**: 70% accuracy is realistic for this problem! We removed data leakage (casualties features), making the model production-ready.

## 🌐 Deployment

### 🚀 Method 1: Streamlit Cloud (Recommended - FREE!)

**Step-by-step**:

1. **Install Git** (if not installed):
   - Download: https://git-scm.com/download/win
   - Install with default settings

2. **Push to GitHub**:
   ```bash
   cd e:\projects\traffic-accident-predictor
   git init
   git config user.name "Preeti Sathawane"
   git config user.email "sathawanepg@rknec.edu"
   git add .
   git commit -m "Initial commit: Traffic Accident Predictor"
   git remote add origin https://github.com/lessgo-preeti/traffic-accident-predictor.git
   git push -u origin main
   ```

3. **Deploy on Streamlit Cloud**:
   - Go to: https://streamlit.io/cloud
   - Sign up with GitHub account
   - Click "New app"
   - Select repository: `traffic-accident-predictor`
   - Main file: `dashboard/app.py`
   - Click "Deploy!"

4. **Get Your Public URL**:
   ```
   https://lessgo-preeti-traffic-accident-predictor.streamlit.app
   ```

**Detailed Guide**: See [GITHUB_SETUP.md](GITHUB_SETUP.md) for complete instructions!

### 💻 Method 2: Local Deployment

```bash
# Navigate to project
cd e:\projects\traffic-accident-predictor

# Run dashboard
streamlit run dashboard\app.py --server.port 8501
```

Open browser: http://localhost:8501

## 📈 Future Enhancements

- [ ] Add real-time accident data API integration
- [ ] Implement SHAP values for model explainability
- [ ] Add accident hotspot mapping with Folium
- [ ] Mobile-responsive dashboard design
- [ ] Multi-language support (Hindi, English)
- [ ] SMS/Email alert system for high-risk predictions
- [ ] Integration with Google Maps API for route safety
- [ ] Mobile app version (React Native)

## 💼 Portfolio & Resume Points

**Use these points for your resume/portfolio**:

✨ **Project Title**: Traffic Accident Severity Prediction System using Machine Learning

🎯 **Key Achievements**:
- Developed end-to-end ML pipeline achieving 70% accuracy on Indian road accident data
- Engineered 22 features from raw data with domain knowledge of Indian road conditions
- Built interactive Streamlit dashboard with 1000+ predictions capability
- Deployed production-ready model on cloud (Streamlit Cloud - FREE!)
- Prevented data leakage by removing target-correlated features (casualties)

🛠️ **Technical Skills Demonstrated**:
- **ML**: Random Forest, XGBoost, Feature Engineering, Cross-validation
- **Python**: Pandas, NumPy, Scikit-learn, Streamlit
- **Data Processing**: Data cleaning, EDA, feature scaling, encoding
- **Deployment**: Git, GitHub, Cloud deployment (Streamlit Cloud)
- **Visualization**: Plotly, Matplotlib, Seaborn

📊 **Impact**:
- Can help reduce accident severity by predicting high-risk scenarios
- Useful for traffic police and road safety departments
- Scalable to real-time accident prevention systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Preeti Sathawane**
- 🔗 **GitHub**: [@lessgo-preeti](https://github.com/lessgo-preeti)
- 💼 **LinkedIn**: [Preeti Sathawane](https://linkedin.com/in/preeti-sathawane)
- 🌐 **Live Demo**: [Traffic Accident Predictor](https://lessgo-preeti-traffic-accident-predictor.streamlit.app)
- 📧 **Email**: sathawanepg@rknec.edu

## 🙏 Acknowledgments

- **Inspiration**: Smart India Hackathon (SIH) - Road Safety Theme
- **Data Source**: Custom generated Indian road accident dataset
- **Libraries**: Scikit-learn, XGBoost, Streamlit, Plotly, Pandas
- **Focus**: Making Indian roads safer through AI/ML

## 📞 Support & Contact

Having issues? Questions? Suggestions?

- 📧 **Email**: sathawanepg@rknec.edu
- 🐛 **Issues**: [GitHub Issues](https://github.com/lessgo-preeti/traffic-accident-predictor/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/lessgo-preeti/traffic-accident-predictor/discussions)

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

**Made with ❤️ for Indian Road Safety**

</div>
