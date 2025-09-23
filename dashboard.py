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

# --- WHAT-IF & ADVANCED ANALYSIS FUNCTIONS ---

@st.cache_data
def generate_frontier_data():
    """Generates the data points for the Impact Frontier graph."""
    # Assumptions from our analysis
    dev_potential = 20 # Using the relative numbers from the final analysis
    dev_ing_potential = 9
    adoption_rates = {'dev_dark': 0.3, 'dev_light': 0.6, 'dev_ing_dark': 0.1, 'dev_ing_light': 0.8}
    quality_scores = {'dark': 9, 'light': 4}
    
    frontier_points = []
    for i in range(11):
        mix_dark = i / 10.0
        mix_light = 1.0 - mix_dark
        
        cap_dev_dark = dev_potential * adoption_rates['dev_dark'] * mix_dark
        cap_dev_light = dev_potential * adoption_rates['dev_light'] * mix_light
        cap_deving_dark = dev_ing_potential * adoption_rates['dev_ing_dark'] * mix_dark
        cap_deving_light = dev_ing_potential * adoption_rates['dev_ing_light'] * mix_light
        
        total_capital = cap_dev_dark + cap_dev_light + cap_deving_dark + cap_deving_light
        
        if total_capital > 0:
            total_quality_points = ((cap_dev_dark + cap_deving_dark) * quality_scores['dark']) + \
                                   ((cap_dev_light + cap_deving_light) * quality_scores['light'])
            avg_quality = total_quality_points / total_capital
        else:
            avg_quality = quality_scores['dark'] if mix_dark == 1.0 else quality_scores['light']
            
        frontier_points.append({'mix': mix_dark * 100, 'TotalCapital': total_capital, 'AverageEQS': avg_quality})
        
    return pd.DataFrame(frontier_points)

@st.cache_data
def run_regression_analysis(df):
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
    if df is None or df.empty: return None
    features = df[['ESGAwareness', 'BiasPrevalence']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features).astype(str)
    return df

# --- MAIN APP ---
df_bonds, df_bias, df_policy = load_data()
st.title("🌿 Sustainable Finance Project Dashboard")
st.markdown("An interactive summary of the key findings and strategic analyses across three core research objectives.")

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
    # ... (Descriptive charts for Obj 1 remain the same)
    with st.expander("What-If Analysis: The Impact Frontier of Framework Design"):
        st.markdown("This model visualizes the strategic trade-off between maximizing total capital raised and ensuring high environmental quality. Use the slider to explore different framework designs.")
        frontier_data = generate_frontier_data()
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("The Impact Frontier")
            fig_frontier = px.scatter(frontier_data, x='TotalCapital', y='AverageEQS',
                                      labels={'TotalCapital': 'Total Capital Mobilised (Relative)', 'AverageEQS': 'Average Environmental Quality Score'},
                                      title='Trade-off between Capital and Quality')
            fig_frontier.update_traces(mode='lines+markers')
            st.plotly_chart(fig_frontier, use_container_width=True)
            
        with col2:
            st.subheader("Test a Policy Mix")
            selected_mix = st.slider("Select Framework Policy Mix (% Dark Green)", 0, 100, 50, 10)
            
            # Find the closest point on the frontier to the slider's value
            closest_point = frontier_data.iloc[(frontier_data['mix'] - selected_mix).abs().argsort()[:1]]
            capital_result = closest_point['TotalCapital'].values[0]
            quality_result = closest_point['AverageEQS'].values[0]

            st.metric("Resulting Capital Mobilised", f"{capital_result:.2f}")
            st.metric("Resulting Average Quality", f"{quality_result:.2f}")
            st.info("The analysis shows that the efficiency of the trade-off (quality gained per unit of capital sacrificed) consistently improves as the framework becomes stricter.")

else: st.warning("Data for Objective 1 could not be loaded or is empty for the selected filters.")

# --- OBJECTIVE 2 ---
st.header("Objective 2: The Investor Psychology Landscape")
if df_bias is not None:
    # ... (All descriptive, regression, and clustering charts for Obj 2 remain the same)
    st.markdown("### Descriptive Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The Awareness-Bias Relationship")
        fig_scatter = px.scatter(df_bias, x='ESGAwareness', y='BiasPrevalence', hover_name='Region', trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        st.subheader("Bias Prevalence by Region")
        fig_bias_bar = px.bar(df_bias.sort_values('BiasPrevalence', ascending=False), x='Region', y='BiasPrevalence')
        st.plotly_chart(fig_bias_bar, use_container_width=True)

    with st.expander("Advanced Analysis: Predictive Modeling & Segmentation"):
        st.markdown("### Advanced Driver Analysis (Predictive Model)")
        model_score, coef_summary = run_regression_analysis(df_bias)
        if model_score is not None:
            # ... (Regression model display)
            col1, col2 = st.columns(2)
            with col1:
                 st.metric("Model R-squared (Accuracy)", f"{model_score:.2f}", help=f"This model explains {model_score:.0%} of the variation in investor bias.")
                 st.subheader("Key Drivers of Investor Bias")
                 coef_summary_chart = coef_summary.sort_values('AbsoluteCoefficient', ascending=True)
                 fig_importance = px.bar(coef_summary_chart, y=coef_summary_chart.index, x='AbsoluteCoefficient', orientation='h')
                 st.plotly_chart(fig_importance, use_container_width=True)
            with col2:
                st.subheader("Detailed Model Coefficients")
                st.dataframe(coef_summary)
                st.success(f"**Conclusion:** The single most important predictor of investor bias is **'{coef_summary.index[0]}'**.")

        st.markdown("---")
        st.markdown("### Advanced Analysis: Investor Segmentation (K-Means Clustering)")
        df_bias_clustered = run_kmeans_clustering(df_bias.copy())
        if df_bias_clustered is not None:
            # ... (Clustering model display)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Investor Segments Visualized")
                fig_cluster = px.scatter(df_bias_clustered, x='ESGAwareness', y='BiasPrevalence', color='Cluster', hover_name='Region')
                st.plotly_chart(fig_cluster, use_container_width=True)
            with col2:
                st.subheader("Cluster Profiles")
                cluster_profiles = df_bias_clustered.groupby('Cluster')[['ESGAwareness', 'BiasPrevalence']].mean().round(2)
                st.dataframe(cluster_profiles)
                st.info("Use the average values to identify segments like 'High-Risk Novice' or 'Informed & Confident'.")

else: st.warning("Data for Objective 2 could not be loaded.")

# --- OBJECTIVE 3 ---
st.header("Objective 3: The Global Biodiversity Policy Toolkit")
if df_policy is not None:
    # ... (Descriptive charts for Obj 3 remain the same)
    st.markdown("### Descriptive Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Policy Mix")
        policy_mix = df_policy['InstrumentType'].value_counts().reset_index()
        fig_pie = px.pie(policy_mix, names='InstrumentType', values='count', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("OECD vs. Non-OECD Policy Toolkits")
        oecd_mix = df_policy.groupby(['OECD_Status', 'InstrumentType']).size().reset_index(name='Count')
        fig_stacked = px.bar(oecd_mix, x='OECD_Status', y='Count', color='InstrumentType', barmode='stack')
        st.plotly_chart(fig_stacked, use_container_width=True)

    with st.expander("What-If Analysis 1: The Policy Maturity Lifecycle"):
        st.markdown("This model shows the long-term benefit of accelerating a country's evolution from simple to sophisticated policy tools.")
        # Timeline data generation
        years = np.arange(1, 21)
        power = {'Phase 1': 50, 'Phase 2': 100, 'Phase 3': 300}
        natural_evo = [power['Phase 1']] * 10 + [power['Phase 2']] * 10
        accel_evo = [power['Phase 1']] * 5 + [power['Phase 2']] * 10 + [power['Phase 3']] * 5
        df_evo = pd.DataFrame({
            'Year': years,
            'Natural Evolution': np.cumsum(natural_evo),
            'Accelerated Evolution': np.cumsum(accel_evo)
        })
        fig_evo = px.line(df_evo, x='Year', y=['Natural Evolution', 'Accelerated Evolution'], title='Cumulative Capital Mobilised Over Time')
        st.plotly_chart(fig_evo, use_container_width=True)

    with st.expander("What-If Analysis 2: Financial Resilience Stress Test"):
        st.markdown("This model stress-tests two different policy portfolios against a government budget crisis.")
        cut_percent = st.slider("Select % Cut to Subsidy Budget", 0, 100, 25, 5)
        
        # Portfolio definitions
        subsidy_reliant_funding = 70 * (1 - cut_percent/100) + 30
        diversified_funding = 20 * (1 - cut_percent/100) + 40 + 40
        
        col1, col2 = st.columns(2)
        col1.metric("Subsidy-Reliant Portfolio Funding", f"${subsidy_reliant_funding:.1f} M")
        col2.metric("Diversified Portfolio Funding", f"${diversified_funding:.1f} M", delta=f"{diversified_funding - subsidy_reliant_funding:.1f} M")
        
        # Data for the graph
        cuts = np.arange(0, 101, 5)
        sr_funding = [70 * (1 - c/100) + 30 for c in cuts]
        div_funding = [20 * (1 - c/100) + 80 for c in cuts]
        df_stress = pd.DataFrame({'% Cut': cuts, 'Subsidy-Reliant': sr_funding, 'Diversified': div_funding})
        
        fig_stress = px.line(df_stress, x='% Cut', y=['Subsidy-Reliant', 'Diversified'], title='Portfolio Funding Under Fiscal Stress')
        st.plotly_chart(fig_stress, use_container_width=True)
        st.info("The analysis shows the diversified portfolio is far more resilient to shocks because it doesn't rely on a single source of funding.")

else: st.warning("Data for Objective 3 could not be loaded.")

