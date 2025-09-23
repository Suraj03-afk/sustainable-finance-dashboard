import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sustainable Finance Dashboard",
    page_icon="🌿",
    layout="wide",
)

# --- DATA LOADING AND CLEANING ---
@st.cache_data
def load_data():
    """Loads data from the three specified Excel workbooks."""
    df_bonds = None
    df_bias = None
    df_policy = None

    # Load Objective 1 Data
    try:
        bonds_filename = 'news_makers_export analysis - Copy.xlsx'
        df_bonds = pd.read_excel(bonds_filename, sheet_name='news_makers_export')
        df_bonds.columns = [str(col).strip() for col in df_bonds.columns]
        df_bonds['Issue Date'] = pd.to_datetime(df_bonds['Issue Date'], errors='coerce')
        df_bonds['Year'] = df_bonds['Issue Date'].dt.year
        df_bonds['Amount (USD)'] = pd.to_numeric(df_bonds['Amount (USD)'], errors='coerce')
        df_bonds.dropna(subset=['Issue Date', 'Amount (USD)', 'Country', 'Sector'], inplace=True)
    except Exception as e:
        st.error(f"Error loading Objective 1 data from `{bonds_filename}`. Please check file and sheet name ('news_makers_export'). Error: {e}")

    # Load Objective 2 Data
    try:
        bias_filename = 'Behavioral_Bias_SRI_Dataset - Copy.xlsx'
        df_bias = pd.read_excel(bias_filename, sheet_name='Sheet2')
        required_cols = ['Region', 'Investor Type', 'ESG Awareness (%)', 'Bias Prevalence (%)']
        if not all(col in df_bias.columns for col in required_cols):
             st.error(f"The sheet 'Sheet2' in `{bias_filename}` is missing required columns.")
        else:
            df_bias = df_bias[required_cols]
            df_bias.columns = ['Region', 'InvestorType', 'ESGAwareness', 'BiasPrevalence']
            df_bias.dropna(inplace=True)
    except Exception as e:
        st.error(f"Error loading Objective 2 data from `{bias_filename}`. Please check file and sheet name ('Sheet2'). Error: {e}")

    # Load Objective 3 Data
    try:
        policy_filename = 'OECD-PINEVersion2025 - Copy.xlsx'
        df_policy = pd.read_excel(policy_filename, sheet_name='OECD-PINEVersion2025 Objective ')
        oecd_countries = ["Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia", "Costa Rica", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Japan", "Korea", "Latvia", "Lithuania", "Luxembourg", "Mexico", "Netherlands", "New Zealand", "Norway", "Poland", "Portugal", "Slovak Republic", "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "United Kingdom", "United States"]
        df_policy['OECD_Status'] = df_policy['CountryName'].apply(lambda x: 'OECD' if x in oecd_countries else 'Non-OECD')
    except Exception as e:
        st.error(f"Error loading Objective 3 data from `{policy_filename}`. Please check file and sheet name ('OECD-PINEVersion2025 Objective '). Error: {e}")

    return df_bonds, df_bias, df_policy

# --- ADVANCED ANALYSIS FUNCTIONS ---
@st.cache_data
def run_regression_analysis(df):
    """
    Performs a multiple linear regression to identify the key drivers of investor bias.
    """
    if df is None or df.empty: return None, None
    X = df[['Region', 'InvestorType', 'ESGAwareness']]
    y = df['BiasPrevalence']
    categorical_features = ['Region', 'InvestorType']
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    model.fit(X, y)
    score = model.score(X, y)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    coefficients = model.named_steps['regressor'].coef_
    coef_summary = pd.DataFrame(coefficients, index=feature_names, columns=['Coefficient'])
    coef_summary['AbsoluteCoefficient'] = np.abs(coef_summary['Coefficient'])
    coef_summary = coef_summary.sort_values('AbsoluteCoefficient', ascending=False)
    coef_summary.index = coef_summary.index.str.replace('remainder__', '').str.replace('cat__', '')
    return score, coef_summary

@st.cache_data
def run_kmeans_clustering(df):
    """
    Performs K-Means clustering to segment investors.
    """
    if df is None or df.empty or 'ESGAwareness' not in df.columns or 'BiasPrevalence' not in df.columns:
        return None
    features = df[['ESGAwareness', 'BiasPrevalence']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features).astype(str)
    return df

# --- MAIN APP LOGIC ---
df_bonds, df_bias, df_policy = load_data()
st.title("🌿 Sustainable Finance Project Dashboard")
st.markdown("An interactive summary of the key findings across three core research objectives.")

# --- SIDEBAR FILTERS ---
if df_bonds is not None:
    st.sidebar.header("Dashboard Filters")
    min_year, max_year = int(df_bonds['Year'].min()), int(df_bonds['Year'].max())
    selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    all_countries = sorted(df_bonds['Country'].astype(str).unique())
    selected_countries = st.sidebar.multiselect("Select Countries", all_countries, default=all_countries)
    df_bonds_filtered = df_bonds[(df_bonds['Year'] >= selected_years[0]) & (df_bonds['Year'] <= selected_years[1]) & (df_bonds['Country'].isin(selected_countries))]
else:
    st.sidebar.warning("Data for filters could not be loaded.")
    df_bonds_filtered = pd.DataFrame()

# --- OBJECTIVE 1: GLOBAL GREEN FINANCE MARKET ---
st.header("Objective 1: The Global Green Finance Market")
if df_bonds is None: st.warning("Data for Objective 1 could not be loaded.")
elif not df_bonds_filtered.empty:
    col1, col2, col3 = st.columns(3)
    total_capital = df_bonds_filtered['Amount (USD)'].sum()
    num_bonds = len(df_bonds_filtered)
    avg_deal_size = total_capital / num_bonds if num_bonds > 0 else 0
    col1.metric("Total Capital Mobilised", f"${total_capital/1e9:.2f} B")
    col2.metric("Total Number of Bonds", f"{num_bonds:,}")
    col3.metric("Average Deal Size", f"${avg_deal_size/1e6:.2f} M")
    c1, c2 = st.columns((6, 4))
    with c1:
        st.subheader("Green Bond Growth Over Time")
        growth_over_time = df_bonds_filtered.groupby('Year')['Amount (USD)'].sum().reset_index()
        fig_growth = px.line(growth_over_time, x='Year', y='Amount (USD)', markers=True, labels={"Amount (USD)": "Capital Mobilised (USD)"})
        st.plotly_chart(fig_growth, use_container_width=True)
    with c2:
        st.subheader("Issuance by Sector")
        sector_dist = df_bonds_filtered['Sector'].value_counts().nlargest(10).reset_index()
        fig_sector = px.pie(sector_dist, names='Sector', values='count', hole=0.4, title="Top 10 Sectors")
        st.plotly_chart(fig_sector, use_container_width=True)
else: st.warning("No data available for the selected filters.")

# --- OBJECTIVE 2: INVESTOR PSYCHOLOGY LANDSCAPE ---
st.header("Objective 2: The Investor Psychology Landscape")
if df_bias is not None:
    st.markdown("### Descriptive Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The Awareness-Bias Relationship")
        fig_scatter = px.scatter(df_bias, x='ESGAwareness', y='BiasPrevalence', hover_name='Region', trendline="ols", labels={"ESGAwareness": "ESG Awareness (%)", "BiasPrevalence": "Bias Prevalence (%)"})
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        st.subheader("Bias Prevalence by Region")
        bias_by_region = df_bias.sort_values('BiasPrevalence', ascending=False)
        fig_bias_bar = px.bar(bias_by_region, x='Region', y='BiasPrevalence', labels={"BiasPrevalence": "Bias Prevalence (%)"})
        st.plotly_chart(fig_bias_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### Advanced Driver Analysis (Predictive Model)")
    model_score, coef_summary = run_regression_analysis(df_bias)
    if model_score is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model R-squared (Accuracy)", f"{model_score:.2f}", help=f"This model explains {model_score:.0%} of the variation in investor bias.")
            st.subheader("Key Drivers of Investor Bias")
            coef_summary_chart = coef_summary.sort_values('AbsoluteCoefficient', ascending=True)
            fig_importance = px.bar(coef_summary_chart, y=coef_summary_chart.index, x='AbsoluteCoefficient', orientation='h', labels={"y": "Feature", "AbsoluteCoefficient": "Impact Strength"})
            st.plotly_chart(fig_importance, use_container_width=True)
        with col2:
            st.subheader("Detailed Model Coefficients")
            st.dataframe(coef_summary)
            st.success(f"**Conclusion:** The single most important predictor of investor bias is **'{coef_summary.index[0]}'**.")
    else: st.warning("Could not run the regression analysis.")

    st.markdown("---")
    st.markdown("### Advanced Analysis: Investor Segmentation (K-Means Clustering)")
    st.write("This unsupervised learning model automatically groups investors into distinct segments based on their awareness and bias levels, revealing natural archetypes in the market.")
    df_bias_clustered = run_kmeans_clustering(df_bias.copy())
    if df_bias_clustered is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Investor Segments Visualized")
            fig_cluster = px.scatter(df_bias_clustered, x='ESGAwareness', y='BiasPrevalence', color='Cluster', hover_name='Region', title="Investor Clusters")
            st.plotly_chart(fig_cluster, use_container_width=True)
        with col2:
            st.subheader("Cluster Profiles")
            cluster_profiles = df_bias_clustered.groupby('Cluster')[['ESGAwareness', 'BiasPrevalence']].mean().round(2)
            st.dataframe(cluster_profiles)
            st.info("**Interpreting the Clusters:** Use the average values above to identify segments like 'High-Risk Novice' (Low Awareness, High Bias) or 'Informed & Confident' (High Awareness, Low Bias).")
    else: st.warning("Could not run the clustering analysis.")
else: st.warning("Data for Objective 2 could not be loaded.")

# --- OBJECTIVE 3: GLOBAL BIODIVERSITY POLICY TOOLKIT ---
st.header("Objective 3: The Global Biodiversity Policy Toolkit")
if df_policy is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Policy Mix")
        policy_mix = df_policy['InstrumentType'].value_counts().reset_index()
        fig_pie = px.pie(policy_mix, names='InstrumentType', values='count', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("OECD vs. Non-OECD Policy Toolkits")
        oecd_mix = df_policy.groupby(['OECD_Status', 'InstrumentType']).size().reset_index(name='Count')
        fig_stacked = px.bar(oecd_mix, x='OECD_Status', y='Count', color='InstrumentType', title="Composition of Policy Toolkits", barmode='stack')
        st.plotly_chart(fig_stacked, use_container_width=True)
else: st.warning("Data for Objective 3 could not be loaded.")

