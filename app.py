import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from mlxtend.frequent_patterns import apriori, association_rules
import folium
from streamlit_folium import folium_static
import base64
from datetime import datetime

# ------------------------------
# Page configuration
st.set_page_config(
    page_title="SmartCharging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a high‑tech look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        color: #00ff88;
        text-align: center;
        text-shadow: 0 0 10px #00ff88;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #cccccc;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,255,136,0.2);
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(145deg, #1a1a1a, #0d0d0d);
        border-left: 5px solid #00ff88;
        padding: 15px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #2d2d2d;
        border-radius: 5px 5px 0 0;
        color: #ffffff;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff88 !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv("detailed_ev_charging_stations.csv")
    return df

@st.cache_data
def preprocess_data(df):
    # Make a copy
    data = df.copy()
    
    # Handle missing values
    # Reviews (Rating) - fill with median
    data['Reviews (Rating)'] = data['Reviews (Rating)'].fillna(data['Reviews (Rating)'].median())
    # Renewable Energy Source - fill with mode
    data['Renewable Energy Source'] = data['Renewable Energy Source'].fillna('No')
    # Connector Types - fill with 'Unknown'
    data['Connector Types'] = data['Connector Types'].fillna('Unknown')
    # Maintenance Frequency - fill with mode
    data['Maintenance Frequency'] = data['Maintenance Frequency'].fillna('Annually')
    
    # Convert Availability to numeric hours
    def availability_to_hours(avail):
        if pd.isna(avail):
            return 0
        if avail == '24/7':
            return 24
        try:
            # format like "9:00-18:00"
            start, end = avail.split('-')
            start_h = int(start.split(':')[0])
            end_h = int(end.split(':')[0])
            return end_h - start_h
        except:
            return 0
    data['Availability Hours'] = data['Availability'].apply(availability_to_hours)
    
    # Encode categorical features
    # Charger Type
    le_charger = LabelEncoder()
    data['Charger Type Enc'] = le_charger.fit_transform(data['Charger Type'])
    
    # Station Operator (keep top 10 most frequent, others as 'Other')
    top_operators = data['Station Operator'].value_counts().nlargest(10).index
    data['Operator Enc'] = data['Station Operator'].apply(lambda x: x if x in top_operators else 'Other')
    le_operator = LabelEncoder()
    data['Operator Enc'] = le_operator.fit_transform(data['Operator Enc'])
    
    # Renewable Energy Source
    data['Renewable Enc'] = data['Renewable Energy Source'].map({'Yes': 1, 'No': 0})
    
    # Maintenance Frequency ordinal
    freq_map = {'Annually': 0, 'Quarterly': 1, 'Monthly': 2}
    data['Maint Freq Enc'] = data['Maintenance Frequency'].map(freq_map)
    
    # Normalize continuous features for clustering
    continuous_cols = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 
                       'Charging Capacity (kW)', 'Distance to City (km)',
                       'Reviews (Rating)', 'Parking Spots', 'Availability Hours']
    scaler = StandardScaler()
    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])
    
    return data, scaler, le_charger, le_operator

# ------------------------------
# Load and preprocess
df_raw = load_data()
df, scaler, le_charger, le_operator = preprocess_data(df_raw)

# ------------------------------
# Sidebar for navigation
st.sidebar.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=80)
st.sidebar.title("⚡ SmartCharging")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Overview", "📊 EDA", "🔍 Clustering", "🔗 Association Rules", "⚠️ Anomaly Detection", "🗺️ Map View", "📈 Insights"])

# ------------------------------
# Helper function to revert scaling for display
def revert_scale(series, col):
    mean = scaler.mean_[continuous_cols.index(col)] if hasattr(scaler, 'mean_') else 0
    scale = scaler.scale_[continuous_cols.index(col)] if hasattr(scaler, 'scale_') else 1
    return series * scale + mean

continuous_cols = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 
                   'Charging Capacity (kW)', 'Distance to City (km)',
                   'Reviews (Rating)', 'Parking Spots', 'Availability Hours']

# ------------------------------
if page == "🏠 Overview":
    st.markdown('<div class="main-header">⚡ SmartCharging Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Uncovering EV Behavior Patterns with AI</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stations", f"{len(df):,}")
    with col2:
        st.metric("Avg Daily Users", f"{df_raw['Usage Stats (avg users/day)'].mean():.1f}")
    with col3:
        st.metric("Avg Cost (USD/kWh)", f"${df_raw['Cost (USD/kWh)'].mean():.2f}")
    
    st.markdown("---")
    st.subheader("📋 Project Scope")
    st.info("""
    **Objective:** Analyze EV charging station data to cluster behaviors, discover associations, detect anomalies, and provide actionable insights for infrastructure planning.
    
    **Dataset:** 5000+ stations worldwide with details on location, charger type, usage, cost, operator, and reviews.
    
    **Approach:** 
    - Data cleaning & preprocessing (handling missing values, encoding, normalization)
    - Exploratory Data Analysis (interactive visualizations)
    - Clustering (K-Means) to identify user profiles
    - Association Rule Mining (Apriori) to find feature relationships
    - Anomaly Detection (IQR, LOF) to spot unusual stations
    - Interactive maps and dashboards
    """)
    
    st.markdown("### 🚀 Key Features")
    st.markdown("""
    - **Interactive EDA** with Plotly charts
    - **K-Means clustering** with elbow method and map visualization
    - **Association rules** between charger type, operator, and usage
    - **Anomaly detection** using statistical and machine learning methods
    - **Live map** with cluster and anomaly overlays
    - **Actionable insights** for decision‑makers
    """)

elif page == "📊 EDA":
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Usage Stats", "Cost Analysis", "Charger Types", "Correlations"])
    
    with tab1:
        st.subheader("Distribution of Average Daily Users")
        fig = px.histogram(df_raw, x='Usage Stats (avg users/day)', nbins=50, 
                           marginal='box', color_discrete_sequence=['#00ff88'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Usage by Charger Type")
        fig = px.box(df_raw, x='Charger Type', y='Usage Stats (avg users/day)', 
                     color='Charger Type', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Cost Distribution")
        fig = px.histogram(df_raw, x='Cost (USD/kWh)', nbins=50, marginal='violin',
                           color_discrete_sequence=['#ffaa00'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Cost vs. Usage")
        fig = px.scatter(df_raw, x='Cost (USD/kWh)', y='Usage Stats (avg users/day)',
                         color='Charger Type', size='Charging Capacity (kW)',
                         hover_data=['Station Operator'], opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Charger Type Popularity")
        charger_counts = df_raw['Charger Type'].value_counts().reset_index()
        charger_counts.columns = ['Charger Type', 'Count']
        fig = px.bar(charger_counts, x='Charger Type', y='Count', color='Charger Type',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top Station Operators")
        top_ops = df_raw['Station Operator'].value_counts().head(10).reset_index()
        top_ops.columns = ['Operator', 'Count']
        fig = px.bar(top_ops, x='Count', y='Operator', orientation='h',
                     color='Count', color_continuous_scale='greens')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Correlation Heatmap (Normalized Features)")
        corr = df[continuous_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Clustering":
    st.markdown('<div class="main-header">Clustering Analysis</div>', unsafe_allow_html=True)
    
    # Select features for clustering
    cluster_features = ['Usage Stats (avg users/day)', 'Charging Capacity (kW)', 
                        'Cost (USD/kWh)', 'Distance to City (km)', 'Reviews (Rating)']
    
    X = df[cluster_features].copy()
    
    # Elbow method
    st.subheader("Elbow Method for Optimal k")
    inertia = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    fig = px.line(x=list(K_range), y=inertia, markers=True, 
                  labels={'x': 'Number of clusters k', 'y': 'Inertia'},
                  title='Elbow Method')
    st.plotly_chart(fig, use_container_width=True)
    
    # User selects k
    k = st.slider("Select number of clusters", min_value=2, max_value=10, value=4)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Cluster centers (revert scaling for interpretation)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features)
    for col in cluster_features:
        centers[col] = revert_scale(centers[col], col)
    
    st.subheader("Cluster Centers (original scale)")
    st.dataframe(centers.style.format("{:.2f}"))
    
    # Visualize clusters in 2D using PCA or t-SNE? Use first two features for simplicity
    st.subheader("Cluster Visualization (Usage vs Cost)")
    fig = px.scatter(df, x=df['Usage Stats (avg users/day)'].apply(lambda x: revert_scale(x, 'Usage Stats (avg users/day)')),
                     y=df['Cost (USD/kWh)'].apply(lambda x: revert_scale(x, 'Cost (USD/kWh)')),
                     color=df['Cluster'].astype(str), 
                     hover_data=['Station ID', 'Charger Type'],
                     title='Clusters by Usage and Cost',
                     labels={'x': 'Avg Users/Day', 'y': 'Cost (USD/kWh)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Map view of clusters (will be in Map View tab)
    st.info("Clusters are also shown on the interactive map (Map View tab).")

elif page == "🔗 Association Rules":
    st.markdown('<div class="main-header">Association Rule Mining</div>', unsafe_allow_html=True)
    
    # Discretize continuous variables into categories
    df_rule = df_raw.copy()
    
    # Get the actual max values to avoid bin edge duplication
    usage_max = df_rule['Usage Stats (avg users/day)'].max()
    cost_max = df_rule['Cost (USD/kWh)'].max()
    dist_max = df_rule['Distance to City (km)'].max()
    
    # Usage categories - add a small epsilon to the last bin to avoid duplicate edges
    usage_bins = [0, 20, 50, 100, usage_max + 0.001]  # Add epsilon to last bin
    usage_labels = ['Low (0-20)', 'Medium (21-50)', 'High (51-100)', f'Very High (100-{usage_max:.0f})']
    df_rule['Usage Cat'] = pd.cut(df_rule['Usage Stats (avg users/day)'], bins=usage_bins, labels=usage_labels, right=False)
    
    # Cost categories
    cost_bins = [0, 0.2, 0.4, 0.6, cost_max + 0.001]
    cost_labels = ['Cheap (0-0.2)', 'Moderate (0.21-0.4)', 'Expensive (0.41-0.6)', f'Very Expensive (0.6-{cost_max:.2f})']
    df_rule['Cost Cat'] = pd.cut(df_rule['Cost (USD/kWh)'], bins=cost_bins, labels=cost_labels, right=False)
    
    # Distance categories
    dist_bins = [0, 5, 15, 30, dist_max + 0.001]
    dist_labels = ['Near (0-5km)', 'Medium (6-15km)', 'Far (16-30km)', f'Very Far (30-{dist_max:.0f}km)']
    df_rule['Distance Cat'] = pd.cut(df_rule['Distance to City (km)'], bins=dist_bins, labels=dist_labels, right=False)
    
    # Display value counts to verify categories
    with st.expander("View Category Distributions"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Usage Categories:")
            st.write(df_rule['Usage Cat'].value_counts().sort_index())
        with col2:
            st.write("Cost Categories:")
            st.write(df_rule['Cost Cat'].value_counts().sort_index())
        with col3:
            st.write("Distance Categories:")
            st.write(df_rule['Distance Cat'].value_counts().sort_index())
    
    # Create transactions - each row becomes a list of features
    feature_cols = ['Charger Type', 'Station Operator', 'Renewable Energy Source', 
                    'Usage Cat', 'Cost Cat', 'Distance Cat']
    
    # Filter to top operators to avoid too many unique items
    top_operators = df_rule['Station Operator'].value_counts().nlargest(10).index
    df_rule['Station Operator Simplified'] = df_rule['Station Operator'].apply(
        lambda x: x if x in top_operators else 'Other Operator'
    )
    
    # Update feature columns with simplified operator
    feature_cols_updated = ['Charger Type', 'Station Operator Simplified', 'Renewable Energy Source', 
                            'Usage Cat', 'Cost Cat', 'Distance Cat']
    
    # Create transactions
    transactions = []
    for _, row in df_rule.iterrows():
        transaction = []
        for col in feature_cols:
elif page == "⚠️ Anomaly Detection":
    st.markdown('<div class="main-header">Anomaly Detection</div>', unsafe_allow_html=True)
    
    method = st.selectbox("Select method", ["IQR on Usage", "Z-Score on Usage", "Local Outlier Factor (multi‑feature)"])
    
    if method == "IQR on Usage":
        usage = df_raw['Usage Stats (avg users/day)']
        Q1 = usage.quantile(0.25)
        Q3 = usage.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies = df_raw[(usage < lower_bound) | (usage > upper_bound)]
        st.success(f"Found {len(anomalies)} anomalies using IQR.")
        
    elif method == "Z-Score on Usage":
        from scipy import stats
        usage = df_raw['Usage Stats (avg users/day)']
        z_scores = np.abs(stats.zscore(usage))
        threshold = st.slider("Z-score threshold", 2.0, 5.0, 3.0)
        anomalies = df_raw[z_scores > threshold]
        st.success(f"Found {len(anomalies)} anomalies with z-score > {threshold}.")
        
    else:  # LOF
        features = ['Usage Stats (avg users/day)', 'Cost (USD/kWh)', 'Charging Capacity (kW)']
        X_lof = df_raw[features].fillna(0)
        lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
        y_pred = lof.fit_predict(X_lof)
        anomalies = df_raw[y_pred == -1]
        st.success(f"Found {len(anomalies)} anomalies using Local Outlier Factor.")
    
    if not anomalies.empty:
        st.subheader("Anomaly Details")
        st.dataframe(anomalies[['Station ID', 'Address', 'Charger Type', 'Usage Stats (avg users/day)', 
                                'Cost (USD/kWh)', 'Reviews (Rating)']])
        
        # Map anomalies
        st.subheader("Anomalies on Map")
        m = folium.Map(location=[anomalies['Latitude'].mean(), anomalies['Longitude'].mean()], zoom_start=4)
        for _, row in anomalies.iterrows():
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                popup=f"{row['Station ID']}<br>Usage: {row['Usage Stats (avg users/day)']}<br>Cost: ${row['Cost (USD/kWh)']}",
                icon=folium.Icon(color='red', icon='bolt', prefix='fa')
            ).add_to(m)
        folium_static(m, width=1000, height=500)
    else:
        st.info("No anomalies detected with current settings.")

elif page == "🗺️ Map View":
    st.markdown('<div class="main-header">Interactive Station Map</div>', unsafe_allow_html=True)
    
    # Ensure clustering has been run in the Clustering tab, otherwise run default
    if 'Cluster' not in df.columns:
        # Run default clustering with k=4 for map display
        X = df[['Usage Stats (avg users/day)', 'Charging Capacity (kW)', 
                'Cost (USD/kWh)', 'Distance to City (km)', 'Reviews (Rating)']].copy()
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
    
    # Color palette
    cluster_colors = ['#00ff88', '#ffaa00', '#ff5555', '#55aaff', '#aa55ff', '#ff55aa', '#55ffaa', '#ffaa55', '#aaff55', '#55aaff']
    
    # Create base map
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=4)
    
    # Add markers (sample for performance)
    sample_size = min(2000, len(df))  # limit to 2000 for speed
    df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
    
    for _, row in df_sample.iterrows():
        # Get original usage for popup
        usage_orig = df_raw.loc[row.name, 'Usage Stats (avg users/day)']
        cost_orig = df_raw.loc[row.name, 'Cost (USD/kWh)']
        cluster_id = row['Cluster']
        color = cluster_colors[cluster_id % len(cluster_colors)]
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=f"<b>{row['Station ID']}</b><br>Usage: {usage_orig:.1f}<br>Cost: ${cost_orig:.2f}<br>Cluster: {cluster_id}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    st.subheader("Stations colored by cluster (sample for performance)")
    folium_static(m, width=1000, height=600)
    
    # Also show a table of cluster counts
    st.subheader("Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

elif page == "📈 Insights":
    st.markdown('<div class="main-header">Key Insights & Recommendations</div>', unsafe_allow_html=True)
    
    # Compute some insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧑‍🤝‍🧑 Most Popular Charger Type")
        top_charger = df_raw['Charger Type'].value_counts().idxmax()
        st.metric("Top Charger", top_charger)
        
        st.markdown("### 💰 Cheapest Operator (avg cost)")
        cheapest_op = df_raw.groupby('Station Operator')['Cost (USD/kWh)'].mean().idxmin()
        st.metric("Cheapest Operator", cheapest_op)
        
        st.markdown("### ⭐ Best Rated Operator (avg rating)")
        best_rated = df_raw.groupby('Station Operator')['Reviews (Rating)'].mean().idxmax()
        st.metric("Best Rated", best_rated)
    
    with col2:
        st.markdown("### 📍 City vs. Rural Demand")
        near_city = df_raw[df_raw['Distance to City (km)'] < 5]['Usage Stats (avg users/day)'].mean()
        far_city = df_raw[df_raw['Distance to City (km)'] > 20]['Usage Stats (avg users/day)'].mean()
        st.metric("Near City (<5km) Avg Users", f"{near_city:.1f}")
        st.metric("Far from City (>20km) Avg Users", f"{far_city:.1f}")
        
        st.markdown("### 🔋 Renewable vs Non‑Renewable")
        ren_usage = df_raw[df_raw['Renewable Energy Source'] == 'Yes']['Usage Stats (avg users/day)'].mean()
        non_ren_usage = df_raw[df_raw['Renewable Energy Source'] == 'No']['Usage Stats (avg users/day)'].mean()
        st.metric("Renewable Avg Users", f"{ren_usage:.1f}")
        st.metric("Non‑Renewable Avg Users", f"{non_ren_usage:.1f}")
    
    st.markdown("---")
    st.markdown("### 🔍 Clustering Insights")
    # If clusters exist
    if 'Cluster' in df.columns:
        cluster_summary = df.groupby('Cluster').agg({
            'Usage Stats (avg users/day)': lambda x: revert_scale(x, 'Usage Stats (avg users/day)').mean(),
            'Cost (USD/kWh)': lambda x: revert_scale(x, 'Cost (USD/kWh)').mean(),
            'Charging Capacity (kW)': lambda x: revert_scale(x, 'Charging Capacity (kW)').mean(),
            'Distance to City (km)': lambda x: revert_scale(x, 'Distance to City (km)').mean(),
        }).round(2)
        cluster_summary.columns = ['Avg Users/Day', 'Avg Cost ($/kWh)', 'Avg Capacity (kW)', 'Avg Distance to City (km)']
        st.dataframe(cluster_summary)
        
        # Label clusters
        st.markdown("#### Suggested Cluster Labels:")
        for i in range(len(cluster_summary)):
            if cluster_summary.loc[i, 'Avg Users/Day'] > cluster_summary['Avg Users/Day'].quantile(0.75):
                label = "⚡ High Demand"
            elif cluster_summary.loc[i, 'Avg Users/Day'] < cluster_summary['Avg Users/Day'].quantile(0.25):
                label = "🪫 Low Demand"
            else:
                label = "🔋 Medium Demand"
            st.write(f"**Cluster {i}:** {label}")
    
    st.markdown("---")
    st.markdown("### 📌 Recommendations")
    st.success("""
    1. **Expand DC Fast Chargers** near city centers to capture high demand.
    2. **Promote renewable‑energy stations** – they show comparable usage and attract eco‑conscious users.
    3. **Adjust pricing** at low‑demand stations (e.g., discounts) to improve utilization.
    4. **Investigate anomalies** (very low usage despite good features) for potential maintenance or marketing.
    5. **Target high‑usage clusters** with loyalty programs and premium services.
    """)
