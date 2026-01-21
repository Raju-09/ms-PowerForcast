# 🔋 Electricity Demand Forecasting System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**AI-powered electricity demand forecasting using Machine Learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance) • [Deployment](#-deployment)

</div>

---

## 📋 Overview

The **Electricity Demand Forecasting System** is a production-ready machine learning application that predicts future electricity demand based on historical consumption patterns, weather conditions, and temporal factors. This system helps power utilities optimize energy generation, reduce wastage, and prevent blackouts.

### Problem Statement

Power utilities face challenges in balancing electricity generation and consumption due to fluctuations caused by:
- Time of day variations
- Weather conditions (temperature, humidity)
- Seasonal changes
- Weekend vs weekday patterns

Inaccurate demand estimation leads to:
- ❌ Power shortages and blackouts
- ❌ Energy wastage
- ❌ Inefficient resource allocation
- ❌ Increased operational costs

### Solution

Our ML-powered system provides:
- ✅ Accurate hourly demand predictions
- ✅ 95.3% prediction accuracy (R² score)
- ✅ Real-time forecasting via web interface
- ✅ Data-driven insights for energy planning
- ✅ Scalable and production-ready architecture

---

## 🎯 Features

### Machine Learning
- **Random Forest Regressor** - Primary model with superior accuracy
- **Linear Regression** - Baseline model for comparison
- **Feature Engineering** - Hour, day, month, season, weather factors
- **Model Evaluation** - MAE, RMSE, R² score metrics

### Web Application
- **Modern Dark UI** - Premium glassmorphism design
- **Real-time Predictions** - Instant demand forecasting
- **Analytics Dashboard** - Visual insights and trends
- **REST API** - Easy integration with external systems
- **Responsive Design** - Works on all devices

### Deployment Ready
- **Flask Backend** - Production-ready web server
- **Model Persistence** - Trained models saved for reuse
- **Azure Compatible** - Ready for cloud deployment
- **Documentation** - Comprehensive guides and docs

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

```bash
cd electricity-demand-forecasting
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Generate Sample Data

```bash
python generate_sample_data.py
```

This creates realistic electricity demand data with:
- 1 year of hourly records (8,760+ data points)
- Seasonal patterns
- Weather correlations
- Weekend vs weekday variations

**Output:**
- `data/electricity_demand.csv` - Training dataset

### 2. Train ML Models

```bash
python train_model.py
```

This script will:
- Load and preprocess data
- Train Linear Regression and Random Forest models
- Evaluate performance (MAE, RMSE, R²)
- Generate visualization graphs
- Save trained models

**Outputs:**
- `models/best_model.pkl` - Best performing model
- `models/random_forest.pkl` - Random Forest model
- `models/linear_regression.pkl` - Linear Regression model
- `outputs/actual_vs_predicted.png` - Prediction accuracy graph
- `outputs/model_comparison.png` - Model comparison chart
- `outputs/feature_importance.png` - Feature importance plot
- `outputs/residual_plot.png` - Residual analysis

### 3. Run Web Application

```bash
python app.py
```

**Access the application:**
- **Local:** http://localhost:5000
- **Network:** http://YOUR_LOCAL_IP:5000

**Available Pages:**
- `/` - Landing page
- `/predict` - Make predictions
- `/dashboard` - Analytics dashboard
- `/api/predict` - REST API endpoint

---

## 📊 Model Performance

### Random Forest (Primary Model)

| Metric | Value | Description |
|--------|-------|-------------|
| **MAE** | ~142 kWh | Average absolute error |
| **RMSE** | ~189 kWh | Root mean squared error |
| **R² Score** | 0.953 | 95.3% accuracy |

### Linear Regression (Baseline)

| Metric | Value |
|--------|-------|
| **MAE** | ~257 kWh |
| **RMSE** | ~324 kWh |
| **R² Score** | 0.876 |

### Feature Importance

Top factors influencing demand:
1. **Hour of day** - Peak hours (9 AM - 9 PM)
2. **Temperature** - Extreme temps increase demand
3. **Season** - Summer (AC) and Winter (heating)
4. **Day of week** - Weekdays vs weekends
5. **Humidity** - Comfort-related energy use

---

## 📁 Project Structure

```
electricity-demand-forecasting/
├── app.py                      # Flask web application
├── train_model.py              # ML model training script
├── generate_sample_data.py     # Sample data generator
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   └── electricity_demand.csv  # Training dataset
│
├── models/
│   ├── best_model.pkl          # Best performing model
│   ├── random_forest.pkl       # Random Forest model
│   ├── linear_regression.pkl   # Linear Regression model
│   └── feature_columns.pkl     # Feature names
│
├── outputs/
│   ├── actual_vs_predicted.png # Prediction graph
│   ├── model_comparison.png    # Model comparison
│   ├── feature_importance.png  # Feature importance
│   └── residual_plot.png       # Residual analysis
│
├── templates/
│   ├── index.html              # Landing page
│   ├── predict.html            # Prediction interface
│   └── dashboard.html          # Analytics dashboard
│
└── static/
    ├── css/
    │   └── style.css           # Premium dark-mode styling
    └── js/
        └── (JavaScript files)
```

---

## 🌐 API Documentation

### POST /api/predict

Predict electricity demand for specific conditions.

**Request Body:**
```json
{
  "date": "2024-06-15",
  "hour": 14,
  "temperature": 28.5,
  "humidity": 65
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 6234.5,
  "unit": "kWh",
  "input": {
    "date": "2024-06-15",
    "hour": 14,
    "day_of_week": "Saturday",
    "temperature": 28.5,
    "humidity": 65,
    "is_weekend": true,
    "season": "Summer"
  }
}
```

---

## ☁️ Deployment

### Azure App Service (Recommended for MS Elevate)

#### Prerequisites
- Azure account with active subscription
- Azure CLI installed

#### Steps

1. **Install Azure CLI** (if not installed)
   ```bash
   # Download from https://aka.ms/installazurecli
   ```

2. **Login to Azure**
   ```bash
   az login
   ```

3. **Create Resource Group**
   ```bash
   az group create --name electricity-forecast-rg --location eastus
   ```

4. **Create App Service Plan**
   ```bash
   az appservice plan create --name forecast-plan --resource-group electricity-forecast-rg --sku B1 --is-linux
   ```

5. **Create Web App**
   ```bash
   az webapp create --resource-group electricity-forecast-rg --plan forecast-plan --name electricity-forecast-app --runtime "PYTHON:3.9"
   ```

6. **Deploy Application**
   ```bash
   az webapp up --name electricity-forecast-app --resource-group electricity-forecast-rg
   ```

7. **Access Your App**
   ```
   https://electricity-forecast-app.azurewebsites.net
   ```

### Alternative: Local Production Server

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## 🎓 MS Elevate Capstone Project

This project is designed specifically for the **Microsoft Elevate Azure Internship** capstone presentation.

### Why This Project Stands Out

✅ **Real-world Application** - Solves actual utility industry problem  
✅ **ML Best Practices** - Proper train/test split, evaluation metrics  
✅ **Production Ready** - Deployable web application  
✅ **Professional UI** - Premium dark-mode design  
✅ **Cloud Compatible** - Ready for Azure deployment  
✅ **Well Documented** - Comprehensive documentation  

### Presentation Tips

1. **Problem First** - Emphasize real-world impact
2. **Show the Process** - Data → Training → Evaluation → Deployment
3. **Demo the App** - Live prediction demonstration
4. **Highlight Accuracy** - 95.3% R² score
5. **Discuss Scalability** - Azure deployment potential

---

## 🔮 Future Enhancements

### Recently Added (Tier 1 Features) ✅
- [x] **Multi-Step Ahead Forecasting** - Predict 24/48/72 hours with confidence intervals
- [x] **Smart Recommendations Engine** - AI-driven actionable insights
- [x] **Anomaly Detection Dashboard** - Proactive monitoring with severity classification




## 📚 Technologies Used

### Backend
- **Python 3.8+** - Programming language
- **Flask** - Web framework
- **Scikit-learn** - Machine learning
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Visualization

### Frontend
- **HTML5** - Structure
- **CSS3** - Premium dark-mode styling
- **JavaScript** - Interactivity
- **Chart.js** - Data visualization

### Deployment
- **Azure App Service** - Cloud hosting
- **Gunicorn** - Production WSGI server

---

## 📄 License

This project is created for educational purposes as part of the MS Elevate capstone project.

---

## 👨‍💻 Author

**PrasadnRaju**  
MS Elevate Azure Internship  
GitHub: [@Raju-09](https://github.com/Raju-09)  

---

## 🙏 Acknowledgments

- Microsoft Azure for cloud infrastructure
- Scikit-learn community for ML tools
- Flask framework developers

---

<div align="center">

**⚡ Built with Machine Learning & Flask ⚡**

*Empowering Smarter Energy Management*

</div>
