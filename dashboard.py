import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from prophet import Prophet
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import holidays

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
        st.error(f"Error loading Objective 2 data from `{bias_filename}`. Error: {e}")

    try:
        policy_filename = 'OECD-PINEVersion2025 - Copy.xlsx'
        df_policy = pd.read_excel(policy_filename, sheet_name='OECD-PINEVersion2025 Objective ')
        oecd_countries = ["Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia", "Costa Rica", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Japan", "Korea", "Latvia", "Lithuania", "Luxembourg", "Mexico", "Netherlands", "New Zealand", "Norway", "Poland", "Portugal", "Slovak Republic", "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "United Kingdom", "United States"]
        df_policy['OECD_Status'] = df_policy['CountryName'].apply(lambda x: 'OECD' if x in oecd_countries else 'Non-OECD')
    except Exception as e:
        st.error(f"Error loading Objective 3 data from `{policy_filename}`. Error: {e}")

    return df_bonds, df_bias, df_policy

# --- ADVANCED ANALYSIS FUNCTIONS ---
@st.cache_data
def run_prophet_forecast(df):
    if df is None or len(df) < 2: return None
    prophet_df = df.groupby('Issue Date')['Amount (USD)'].sum().reset_index()
    prophet_df.columns = ['ds', 'y']
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)
    return forecast

@st.cache_data
def run_all_regressions(df):
    if df is None or df.empty: return None, None
    X = df[['Region', 'InvestorType', 'ESGAwareness']]
    y = df['BiasPrevalence']
    categorical_features = ['Region', 'InvestorType']
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression (alpha=1.0)": Ridge(alpha=1.0),
        "Lasso Regression (alpha=0.1)": Lasso(alpha=0.1)
    }
    
    scores = {}
    coefficients = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X, y)
        scores[name] = pipeline.score(X, y)
        
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        coefs = pipeline.named_steps['regressor'].coef_
        
        coef_summary = pd.DataFrame(coefs, index=feature_names, columns=['Coefficient'])
        coef_summary.index = coef_summary.index.str.replace('remainder__', '').str.replace('cat__', '')
        coefficients[name] = coef_summary.sort_values('Coefficient', ascending=False)
        
    return scores, coefficients


@st.cache_data
def run_kmeans_clustering(df):
    if df is None or df.empty: return None
    features = df[['ESGAwareness', 'BiasPrevalence']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features).astype(str)
    return df

@st.cache_data
def run_association_rules(df):
    if df is None or df.empty: return None
    basket = (df.groupby(['CountryName', 'InstrumentType_Detail'])['InstrumentId'].count().unstack().reset_index().fillna(0).set_index('CountryName'))
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
    if frequent_itemsets.empty: return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# --- MAIN APP ---
df_bonds, df_bias, df_policy = load_data()
st.title("🌿 Sustainable Finance Project Dashboard")
st.markdown("An interactive summary of key findings and advanced data analytics across three core research objectives.")

# --- SIDEBAR FILTERS ---
if df_bonds is not None:
    st.sidebar.header("Dashboard Filters")
    min_year, max_year = int(df_bonds['Year'].min()), int(df_bonds['Year'].max())
    selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    all_countries = sorted(df_bonds['Country'].astype(str).unique())
    selected_countries = st.sidebar.multiselect("Select Countries", all_countries, default=all_countries)
    df_bonds_filtered = df_bonds[(df_bonds['Year'] >= selected_years[0]) & (df_bonds['Year'] <= selected_years[1]) & (df_bonds['Country'].isin(selected_countries))]
else:
    df_bonds_filtered = pd.DataFrame()

# --- OBJECTIVE 1 ---
st.header("Objective 1: The Global Green Finance Market")
if not df_bonds_filtered.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Capital Mobilised", f"${df_bonds_filtered['Amount (USD)'].sum()/1e9:.2f}B")
    col2.metric("Number of Bonds", f"{len(df_bonds_filtered):,}")
    col3.metric("Number of Countries", f"{df_bonds_filtered['Country'].nunique()}")
    
    with st.expander("Advanced Analysis: Time Series Forecasting"):
        st.markdown("This model uses the Prophet forecasting library to project the future growth of the green bond market based on historical trends.")
        forecast_data = run_prophet_forecast(df_bonds) # Use unfiltered data for a stable forecast
        if forecast_data is not None:
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], name='Forecast', line=dict(color='royalblue', width=2)))
            fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_upper'], fill=None, mode='lines', line=dict(color='lightgrey'), name='Upper Bound'))
            fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_lower'], fill='tonexty', mode='lines', line=dict(color='lightgrey'), name='Lower Bound'))
            actuals_df = df_bonds.groupby(pd.Grouper(key='Issue Date', freq='M'))['Amount (USD)'].sum().reset_index()
            fig_forecast.add_trace(go.Scatter(x=actuals_df['Issue Date'], y=actuals_df['Amount (USD)'], mode='markers', name='Historical Monthly Data', marker=dict(color='red', size=4)))
            fig_forecast.update_layout(title="5-Year Green Bond Market Forecast", xaxis_title="Date", yaxis_title="Capital Mobilised (USD)")
            st.plotly_chart(fig_forecast, use_container_width=True)
            st.info("The forecast shows the expected growth trajectory and the 'cone of uncertainty' representing the likely range of future outcomes, making a strong case for the urgent need for a standardized framework.")
        else:
            st.warning("Could not generate forecast. More data points are needed.")
else: st.warning("Data for Objective 1 could not be loaded or is empty for the selected filters.")

# --- OBJECTIVE 2 ---
st.header("Objective 2: The Investor Psychology Landscape")
if df_bias is not None:
    st.markdown("### Descriptive Analysis: The Awareness-Bias Relationship")
    fig_scatter = px.scatter(df_bias, x='ESGAwareness', y='BiasPrevalence', hover_name='Region', trendline="ols",
                           labels={"ESGAwareness": "ESG Awareness (%)", "BiasPrevalence": "Bias Prevalence (%)"},
                           title="Higher ESG Awareness is Correlated with Lower Investor Bias")
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("Advanced Analysis: Predictive Modeling & Segmentation"):
        st.markdown("#### 1. Comparing Predictive Models: Linear, Ridge & Lasso")
        st.info("This analysis compares three regression models to predict investor bias. Ridge and Lasso are 'regularized' models that can prevent overfitting and perform feature selection.")
        
        scores, coefficients = run_all_regressions(df_bias)
        
        if scores:
            st.subheader("Model Performance (R-squared)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Linear Regression", f"{scores['Linear Regression']:.3f}")
            col2.metric("Ridge Regression", f"{scores['Ridge Regression (alpha=1.0)']:.3f}")
            col3.metric("Lasso Regression", f"{scores['Lasso Regression (alpha=0.1)']:.3f}")

            st.subheader("Model Coefficients")
            st.markdown("Coefficients show the impact of each feature on bias. Lasso is notable for shrinking unimportant feature coefficients to zero.")
            
            tab1, tab2, tab3 = st.tabs(["Linear Regression", "Ridge Regression", "Lasso Regression"])
            with tab1:
                st.dataframe(coefficients["Linear Regression"])
            with tab2:
                st.dataframe(coefficients["Ridge Regression (alpha=1.0)"])
            with tab3:
                st.dataframe(coefficients["Lasso Regression (alpha=0.1)"])


        st.markdown("---")
        st.markdown("#### 2. What natural investor segments exist in the data?")
        df_bias_clustered = run_kmeans_clustering(df_bias.copy())
        if df_bias_clustered is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Investor Segments Visualized")
                fig_cluster = px.scatter(df_bias_clustered, x='ESGAwareness', y='BiasPrevalence', color='Cluster', hover_name='Region', title="Data-Driven Investor Archetypes")
                st.plotly_chart(fig_cluster, use_container_width=True)
            with col2:
                st.subheader("Cluster Profiles (Average Values)")
                cluster_profiles = df_bias_clustered.groupby('Cluster')[['ESGAwareness', 'BiasPrevalence']].mean().round(2)
                st.dataframe(cluster_profiles)
                st.info("Use these profiles to define data-driven archetypes like 'High-Risk Novice' (low awareness, high bias) or 'Informed & Confident' (high awareness, low bias) for targeted strategies.")
else: st.warning("Data for Objective 2 could not be loaded.")

# --- OBJECTIVE 3 ---
st.header("Objective 3: The Global Biodiversity Policy Toolkit")
if df_policy is not None:
    st.markdown("### Descriptive Analysis: The Current Policy Landscape")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Policy Mix")
        policy_mix = df_policy['InstrumentType'].value_counts().reset_index()
        fig_pie = px.pie(policy_mix, names='InstrumentType', values='count', hole=0.4, title="Current Toolkit is Dominated by Subsidies & Taxes")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("OECD vs. Non-OECD Policy Toolkits")
        oecd_mix = df_policy.groupby(['OECD_Status', 'InstrumentType']).size().reset_index(name='Count')
        fig_stacked = px.bar(oecd_mix, x='OECD_Status', y='Count', color='InstrumentType', barmode='stack', title="Developed Nations Use More Diverse Toolkits")
        st.plotly_chart(fig_stacked, use_container_width=True)

    with st.expander("Advanced Analysis: Discovering the 'Policy Playbook' (Association Rule Mining)"):
        st.markdown("This model finds 'if-then' rules in how countries combine different policy instruments.")
        rules = run_association_rules(df_policy)
        if rules is not None and not rules.empty:
            col1, col2 = st.columns(2)
            min_confidence = col1.slider("Minimum Confidence", 0.0, 1.0, 0.6, 0.05, key="confidence_slider")
            min_lift = col2.slider("Minimum Lift", 0.0, 10.0, 1.5, 0.1, key="lift_slider")
            filtered_rules = rules[(rules['confidence'] >= min_confidence) & (rules['lift'] >= min_lift)]
            st.dataframe(filtered_rules)
            st.info("These rules suggest a potential 'policy playbook' or a natural sequence for adopting new instruments, providing a data-driven roadmap for developing nations.")
        else:
            st.warning("No significant association rules found. Try lowering the filter thresholds.")
else: st.warning("Data for Objective 3 could not be loaded.")

