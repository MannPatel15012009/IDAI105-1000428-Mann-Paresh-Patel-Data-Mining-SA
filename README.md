# ⚡ SmartCharging Analytics

## 📌 Brief Project Title and Scope
SmartCharging Analytics is a data mining project that analyzes electric vehicle (EV) charging station data from over 5,000 global charging stations. The project uses machine learning, statistical analysis, clustering, association rule mining, and anomaly detection to discover patterns in charging usage, pricing strategies, infrastructure efficiency, and station performance.

The project also includes an interactive Streamlit dashboard for visualization, clustering insights, anomaly detection, and infrastructure planning recommendations.

## 🧹 Key Preprocessing Steps, Visualizations, and Findings
### 🔧 Data Preprocessing

The dataset was cleaned and prepared before analysis.

#### Key Preprocessing Steps
Missing value handling using median and mode imputation

##### Feature engineering:

Availability hours

Cost categories

Encoding categorical variables using Label Encoding

Feature scaling using StandardScaler

Removing duplicates

Formatting geographic and numeric data

These preprocessing steps ensured the dataset was ready for machine learning and statistical analysis.

### 📊 Visualizations Used

The following visualizations were used during Exploratory Data Analysis (EDA):

📉 Histograms → Usage distribution, pricing distribution

🔵 Scatter plots → Cost vs usage, capacity vs usage

🔥 Correlation heatmaps → Feature relationships

📊 Bar charts → Charger types and operators

🌍 Geographic maps → Charging station locations

📈 PCA plots → Cluster visualization

🔍 Key Findings from EDA

#### Important insights discovered during EDA:

Stations near city centers have higher usage

DC Fast Chargers are used more frequently

Lower cost per kWh → Higher usage

Renewable energy stations charge slightly higher prices

Some stations have high capacity but low usage

Peak usage occurs in evening hours

These findings were later used in clustering and association rule mining.

## 🤖 Main Insights, Clustering, and Association Techniques Applied

### 🧠 Clustering Analysis
Clustering was performed using K-Means clustering to group charging stations based on usage, cost, capacity, and other features.

#### Methods Used

K-Means clustering

Elbow method

Silhouette score

PCA for visualization

#### Clusters Identified

🏙️ High-demand urban stations

🏡 Budget suburban stations

🌄 Premium rural stations

⚠️ Underutilized stations

Clustering helps identify different types of charging stations and optimize pricing and infrastructure planning.

### 🔗 Association Rule Mining

Association rule mining was performed using the Apriori algorithm.

#### Metrics Used

Support

Confidence

Lift

#### Association Insights
Low-cost stations → High usage probability

DC fast chargers → High daily users

Renewable energy stations → Higher pricing category

Urban stations → High usage clusters

Association rules helped identify relationships between pricing, charger types, renewable energy, and station demand.

### ⚠️ Anomaly Detection

Anomaly detection was performed to identify unusual stations.

#### Methods Used

IQR Method

Z-Score Method

Local Outlier Factor (LOF)

#### Anomaly 

##### 127 anomalies detected (~2.5%)

High price but low usage

Low price but low usage

High capacity but underutilized stations

These stations may require pricing changes or relocation.

## 🌐 Deployed Project on Streamlit, Overview and Functionality
### 🖥️ Streamlit Application Overview

The SmartCharging Analytics project was deployed using Streamlit as an interactive dashboard.

The application allows users to explore EV charging station data through interactive visualizations, clustering dashboards, anomaly detection reports, and geographic maps.

### ⚙️ Streamlit Application Functionality

The Streamlit dashboard allows users to:

📊 Explore usage statistics

💲 Analyze pricing vs usage

🤖 View clustering results

🔗 Explore association rules

⚠️ Identify anomalies

🗺️ View charging stations on interactive maps

🔎 Filter stations by charger type, operator, cost, etc.

The dashboard makes the analysis interactive, visual, and user-friendly.

## 🚀 Deployment Instructions
### 💻 Local Deployment
```bash
git clone https://github.com/MannPatel15012009/IDAI105-1000428-Mann-Paresh-Patel-Data-Mining-SA.git
cd IDAI105-1000428-Mann-Paresh-Patel-Data-Mining-SA

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```
App runs on:

`http://localhost:8501`
### ☁️ Streamlit Cloud Deployment

#### Steps:

Push project to GitHub

Go to Streamlit Cloud

Sign in with GitHub

Click New App

Select repository

Select branch (main)

Enter file path app.py

Click Deploy

## 📚 References
Scikit-learn Documentation
Streamlit Documentation
Plotly Documentation
MLxtend Documentation
Data Mining: Concepts and Techniques — Han, Kamber, Pei
Introduction to Data Mining — Tan, Steinbach, Kumar

## 🖼️ Project Visualizations
### Preprocessing Summary
<img width="1324" height="483" alt="Screenshot 2026-03-27 180826" src="https://github.com/user-attachments/assets/69f2c8e7-7d30-4c66-afd4-59bf9cefc6b5" />

### EDA Dashboard 
<img width="1315" height="615" alt="Screenshot 2026-03-27 181057" src="https://github.com/user-attachments/assets/2a37e2ad-b7f0-4c58-bb5c-f6da0c28dc94" />
<img width="1759" height="804" alt="Screenshot 2026-03-27 181005" src="https://github.com/user-attachments/assets/451e4b0f-c341-47c1-be81-93e3dc386079" />

### Elbow Method
<img width="1395" height="763" alt="Screenshot 2026-03-27 181146" src="https://github.com/user-attachments/assets/e39c2be3-7cfb-4609-8f3f-73006b02cce6" />

### Cluster PCA
<img width="1345" height="682" alt="Screenshot 2026-03-27 181456" src="https://github.com/user-attachments/assets/6a093c8c-854b-4c8f-8da5-076b57c65155" />

### Anomaly Detection
<img width="1405" height="562" alt="Screenshot 2026-03-27 181820" src="https://github.com/user-attachments/assets/c196d443-6926-442f-8a9b-6ac2d1631d1e" />
<img width="1351" height="594" alt="Screenshot 2026-03-27 181836" src="https://github.com/user-attachments/assets/94b79f9a-0a8c-4b72-aafe-7525ed85c668" />

### Interactive Map
<img width="663" height="553" alt="Screenshot 2026-03-27 182038" src="https://github.com/user-attachments/assets/c77696bf-4359-4513-9e7a-34a918e6f269" />

### Insights Dashboar
<img width="1438" height="582" alt="Screenshot 2026-03-27 182119" src="https://github.com/user-attachments/assets/53dc067d-d1fb-4f51-ba0c-7ac4fe9373b0" />
<img width="1182" height="751" alt="Screenshot 2026-03-27 182134" src="https://github.com/user-attachments/assets/226189e2-d7a9-4cf1-b493-9feb081fec6c" />
<img width="1326" height="699" alt="Screenshot 2026-03-27 182222" src="https://github.com/user-attachments/assets/9e1edee6-11d4-4d86-81cd-cf08e994287a" />
### 📁 Repository Structure Guide
```text
SmartCharging-Analytics/
│
├── app.py
├── requirements.txt
├── detailed_ev_charging_stations.csv
├── README.md
│
├── images/
│   ├── preprocessing_summary.png
│   ├── eda_dashboard.png
│   ├── elbow_method.png
│   ├── cluster_pca.png
│   ├── association_rules.png
│   ├── anomaly_map.png
│   ├── interactive_map.png
│   └── insights.png
```
## 👨‍💻 Developer Details

Student Name: Mann Paresh Patel
Candidate Number: 1000428
Course: Data Mining
Year: 2026
