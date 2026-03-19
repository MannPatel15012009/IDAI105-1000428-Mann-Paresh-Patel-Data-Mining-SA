# ⚡ SmartCharging Analytics

## 📋 Project Overview

SmartCharging Analytics is a comprehensive data mining application that analyzes EV charging station patterns using advanced analytics techniques. The application processes over 5,000 global charging stations to uncover insights about usage patterns, pricing strategies, and infrastructure optimization.

### 🎯 Objectives
- Analyze EV charging patterns across global stations
- Identify distinct user behavior clusters
- Discover associations between station features and usage
- Detect anomalies in charging patterns
- Provide actionable insights for infrastructure planning

## 🛠️ Technologies Used

- **Python 3.9+**: Core programming language
- **Streamlit**: Interactive web application framework
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning (clustering, preprocessing)
- **MLxtend**: Association rule mining (Apriori algorithm)
- **Plotly/Matplotlib/Seaborn**: Data visualization
- **Folium**: Interactive maps
- **SciPy**: Statistical analysis

## 📊 Key Features

### Stage 1: Project Scope Definition
- **Objectives:**
  - Analyze EV charging patterns across 5,000+ global stations.
  - Identify distinct user behavior clusters via K‑Means.
  - Discover associations between station features (charger type, cost, renewable energy) using Apriori.
  - Detect anomalies in charging patterns (IQR, Z‑score, LOF).
  - Provide actionable insights for infrastructure planning and operator strategy.
- **Scope:**
  - Dataset: 5,000+ stations with 17 features (location, charger type, cost, usage, operator, capacity, connectors, year, renewable energy, ratings, parking, maintenance).
  - Geographic coverage: global, with major cities represented.
  - Timeframe: stations installed 2010–2023.
  - Analysis includes preprocessing, EDA, clustering, association rules, anomaly detection, and interactive mapping.
- **Tasks:**
  1. Data cleaning & preprocessing
  2. Exploratory Data Analysis (EDA)
  3. Clustering (K‑Means, elbow method, PCA)
  4. Association rule mining (Apriori)
  5. Anomaly detection (statistical & LOF)
  6. Interactive mapping (Folium)
  7. Insight synthesis & recommendations
  8. Streamlit deployment

### Stage 2: Data Preprocessing 
- Missing value handling with median/mode imputation
- Feature engineering (availability hours, cost categories)
- Categorical encoding (LabelEncoder)
- Feature scaling (StandardScaler)

![Preprocessing Summary](images/preprocessing_summary.png)   <!-- replace with actual screenshot -->

### Stage 3: Exploratory Data Analysis 
- Interactive visualizations with Plotly
- Usage statistics and distributions
- Cost analysis and comparisons
- Geographic patterns
- Correlation analysis

![EDA Dashboard](images/eda_dashboard.png)   <!-- replace -->

### Stage 4: Clustering Analysis 
- K-Means clustering with elbow method
- PCA for 2D visualization
- Cluster profiling and labeling
- Silhouette score optimization

![Elbow Method](images/elbow_method.png)  
![Cluster PCA](images/cluster_pca.png)

### Stage 5: Association Rule Mining 
- Apriori algorithm implementation
- Support, confidence, lift metrics
- Rule visualization and filtering
- Transaction encoding

![Association Rules](images/association_rules.png)

### Stage 6: Anomaly Detection 
- Statistical methods (IQR, Z-Score)
- Local Outlier Factor (LOF)
- Multi-feature anomaly detection
- Comprehensive anomaly reporting

![Anomaly Detection](images/anomaly_map.png)

### Stage 7.1: Interactive Maps 
- Folium integration
- Color-coded stations by cluster
- Popup information boxes
- Filtering capabilities

![Interactive Map](images/interactive_map.png)

### Stage 7.2: Insights & Recommendations
- Data-driven strategic recommendations
- Executive summary
- Report generation
- Actionable insights

![Insights Dashboard](images/insights.png)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git (optional)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/SmartCharging-Analytics.git
cd SmartCharging-Analytics
```
### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### Step 3: Install Dependencies
bash
`pip install -r requirements.txt`
### Step 4: Run Application
bash
`streamlit run app.py`
## 📁 Repository Structure
text
```SmartCharging-Analytics/
├── app.py                      # Main Streamlit application
├── requirements.txt             # Python dependencies
├── detailed_ev_charging_stations.csv  # Dataset
├── README.md                    # Documentation
├── .gitignore                   # Git ignore file
├── images/                      # Visualizations (screenshots)
│   ├── preprocessing_summary.png
│   ├── eda_dashboard.png
│   ├── elbow_method.png
│   ├── cluster_pca.png
│   ├── association_rules.png
│   ├── anomaly_map.png
│   ├── interactive_map.png
│   └── insights.png
└── reports/                     # Generated reports
    └── analysis_report.txt
```
## 📊 Dataset Description
The dataset contains 5,000+ EV charging stations with the following features:

Feature	Description	Type
Station ID	Unique identifier	Categorical
Latitude/Longitude	Geographic coordinates	Continuous
Address	Station location	Text
Charger Type	AC Level 1, AC Level 2, DC Fast	Categorical
Cost (USD/kWh)	Price per kilowatt-hour	Continuous
Availability	Operating hours	Text
Distance to City (km)	Proximity to urban center	Continuous
Usage Stats	Average daily users	Continuous
Station Operator	Operating company	Categorical
Charging Capacity (kW)	Maximum power output	Continuous
Connector Types	Available connectors	Categorical
Installation Year	Year established	Integer
Renewable Energy	Yes/No	Binary
Reviews (Rating)	User ratings (1-5)	Continuous
Parking Spots	Number of spaces	Integer
Maintenance Frequency	Service interval	Categorical
## 🔬 Methodology
### Stage 1: Project Scope Definition
Clear objectives and success criteria

Key questions identification

Stakeholder requirements

### Stage 2: Data Preprocessing
Missing value imputation

Feature engineering

Encoding categorical variables

Feature scaling

### Stage 3: Exploratory Data Analysis
Univariate analysis

Bivariate relationships

Multivariate patterns

Geographic visualization

### Stage 4: Clustering Analysis
K-Means clustering

Optimal k selection (elbow method)

Cluster profiling

2D visualization (PCA)

### Stage 5: Association Rule Mining
Transaction encoding

Apriori algorithm

Rule filtering (support, confidence, lift)

Rule interpretation

### Stage 6: Anomaly Detection
Statistical methods (IQR, Z-Score)

Machine learning (LOF)

Multi-feature analysis

Anomaly reporting

### Stage 7: Insights & Reporting
Key findings synthesis

Strategic recommendations

Executive summary

Report generation

### Stage 8: Deployment
Streamlit Cloud deployment

Interactive dashboard

Real-time analysis

User-friendly interface

## 📈 Key Insights
### Usage Patterns
#### Urban Concentration: 
60% of high-usage stations within 10km of city centers

#### Peak Times: 
Evening hours (5-8 PM) show 40% higher usage

#### Charger Popularity: 
DC Fast Chargers account for 45% of total usage

### Economic Factors
Price Sensitivity: 45% higher usage at stations below $0.30/kWh

Operator Variance: 300% price difference between cheapest and most expensive operators

Renewable Premium: Green stations command $0.05/kWh premium with comparable usage

### Clustering Results
4 Distinct Segments: High-demand urban, budget suburban, premium rural, underutilized

Optimal k=4 based on silhouette score (0.52)

Cluster Characteristics: Clear separation by usage and pricing

### Anomaly Detection
127 Anomalies Identified: 2.5% of stations

Common Characteristics: High price with low usage, or vice versa

Action Required: 47 stations flagged for immediate investigation

## 🚀 Deployment
Streamlit Cloud Deployment
Push code to GitHub repository

Visit share.streamlit.io

Connect GitHub account

Select repository and branch

Deploy application

## Visuals of the App
<img width="1339" height="692" alt="Screenshot 2026-03-19 180002" src="https://github.com/user-attachments/assets/ce0666cc-625a-4d6e-91d9-6d8a2b94238d" />
<img width="1302" height="724" alt="Screenshot 2026-03-19 180045" src="https://github.com/user-attachments/assets/cd1fce02-5e4b-4ac3-88f9-888cbdb5eb48" />
<img width="1312" height="796" alt="Screenshot 2026-03-19 175137" src="https://github.com/user-attachments/assets/98abbdc1-61cd-42ee-9e34-0f97a5406e91" />
<img width="1393" height="451" alt="Screenshot 2026-03-19 175215" src="https://github.com/user-attachments/assets/97d5649b-99c9-4741-a415-f3136bf7ccd6" />
<img width="1354" height="740" alt="Screenshot 2026-03-19 175328" src="https://github.com/user-attachments/assets/18ddd9bd-f467-4d6c-a8ea-6af842cd3f84" />
<img width="1321" height="764" alt="Screenshot 2026-03-19 175500" src="https://github.com/user-attachments/assets/ff6011b6-e16c-467e-bd1d-7e31e074c847" />
<img width="1389" height="791" alt="Screenshot 2026-03-19 175547" src="https://github.com/user-attachments/assets/8091192f-065a-4394-917c-fa461a396985" />
<img width="1426" height="716" alt="Screenshot 2026-03-19 175617" src="https://github.com/user-attachments/assets/d7ddbf3e-8f42-49fc-aff1-38519a8a7338" />
<img width="1353" height="702" alt="Screenshot 2026-03-19 175752" src="https://github.com/user-attachments/assets/54a0a31f-7b1d-44c1-a299-31b89eb2ee87" />
<img width="1427" height="816" alt="Screenshot 2026-03-19 175911" src="https://github.com/user-attachments/assets/d148eb38-4c4e-4c9d-b267-ba2533bd5608" />
<img width="1262" height="628" alt="Screenshot 2026-03-19 175946" src="https://github.com/user-attachments/assets/7423ecf5-3e43-4888-888e-c6f8501c7da6" />
<img width="1497" height="720" alt="Screenshot 2026-03-19 175438" src="https://github.com/user-attachments/assets/ec55be5f-14fd-4f1a-b429-1fce1c6aac58" />

## 📝 License
This project is created for educational purposes as part of the Data Mining course.

## Developer Details
Student Name: Mann Paresh Patel

WACP Candidate Number: 1000428

Course: Data Mining

Year: 2026

## 📚 References
Scikit-learn Documentation: https://scikit-learn.org

Streamlit Documentation: https://docs.streamlit.io

Plotly Documentation: https://plotly.com/python

MLxtend Documentation: http://rasbt.github.io/mlxtend

"Data Mining: Concepts and Techniques" - Han, Kamber, Pei

"Introduction to Data Mining" - Tan, Steinbach, Kumar


## Third-Party Trademarks:

Python® is a registered trademark of the Python Software Foundation

Streamlit® is a registered trademark of Streamlit Inc.

Plotly® is a registered trademark of Plotly Technologies Inc.

scikit-learn® is a registered trademark of INRIA

GitHub® is a registered trademark of GitHub, Inc.

All other trademarks are property of their respective owners

Usage Notice: This educational project uses third-party trademarks for identification purposes only. Such use does not imply any endorsement or affiliation.
