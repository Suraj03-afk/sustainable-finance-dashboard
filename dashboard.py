import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sustainable Finance Dashboard",
    page_icon="🌿",
    layout="wide",
)

# --- DATA LOADING AND CLEANING ---
@st.cache_data
def load_data():
    # Load Objective 1 Data from the specific CSV file
    try:
        bonds_filename = 'news_makers_export analysis - Copy.xlsx - news_makers_export.csv'
        df_bonds = pd.read_csv(bonds_filename)
        # Clean column names by removing leading/trailing spaces
        df_bonds.columns = [str(col).strip() for col in df_bonds.columns]
        df_bonds['Issue Date'] = pd.to_datetime(df_bonds['Issue Date'], errors='coerce')
        df_bonds['Year'] = df_bonds['Issue Date'].dt.year
        df_bonds['Amount (USD)'] = pd.to_numeric(df_bonds['Amount (USD)'], errors='coerce')
        df_bonds.dropna(subset=['Issue Date', 'Amount (USD)', 'Country', 'Sector'], inplace=True)
    except FileNotFoundError:
        st.error(f"Error: `{bonds_filename}` not found. Please ensure this file is in the same folder as the script.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading `{bonds_filename}`: {e}")
        return None, None, None

    # Load Objective 2 Data from the specific CSV file
    try:
        bias_filename = 'Behavioral_Bias_SRI_Dataset - Copy.xlsx - Sheet2.csv'
        df_bias = pd.read_csv(bias_filename)
        # Ensure the required columns exist before trying to use them
        required_bias_cols = ['Region', 'Bias Prevalence (%)', 'ESG Awareness (%)']
        if not all(col in df_bias.columns for col in required_bias_cols):
            st.error(f"The file `{bias_filename}` is missing one or more required columns: {required_bias_cols}")
            return None, None, None
        df_bias = df_bias[required_bias_cols]
        df_bias.columns = ['Region', 'BiasPrevalence', 'ESGAwareness']
    except FileNotFoundError:
        st.error(f"Error: `{bias_filename}` not found. Please ensure this file is in the same folder as the script.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading `{bias_filename}`: {e}")
        return None, None, None

    # Load Objective 3 Data from the specific CSV file
    try:
        policy_filename = 'OECD-PINEVersion2025 - Copy.xlsx - OECD-PINEVersion2025 Objective .csv'
        df_policy = pd.read_csv(policy_filename)
        oecd_countries = [
            "Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia", "Costa Rica", "Czech Republic", "Denmark",
            "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Israel", "Italy",
            "Japan", "Korea", "Latvia", "Lithuania", "Luxembourg", "Mexico", "Netherlands", "New Zealand", "Norway",
            "Poland", "Portugal", "Slovak Republic", "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey",
            "United Kingdom", "United States"
        ]
        df_policy['OECD_Status'] = df_policy['CountryName'].apply(lambda x: 'OECD' if x in oecd_countries else 'Non-OECD')
    except FileNotFoundError:
        st.error(f"Error: `{policy_filename}` not found. Please ensure this file is in the same folder as the script.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading `{policy_filename}`: {e}")
        return None, None, None

    return df_bonds, df_bias, df_policy

df_bonds, df_bias, df_policy = load_data()

# --- MAIN DASHBOARD ---
st.title("🌿 Sustainable Finance Project Dashboard")
st.markdown("An interactive summary of the key findings across three core research objectives.")

# --- SIDEBAR FILTERS ---
if df_bonds is not None:
    st.sidebar.header("Dashboard Filters")
    # Year Filter
    min_year, max_year = int(df_bonds['Year'].min()), int(df_bonds['Year'].max())
    selected_years = st.sidebar.slider(
        "Select Year Range",
        min_year, max_year, (min_year, max_year)
    )

    # Country Filter
    all_countries = sorted(df_bonds['Country'].astype(str).unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        all_countries,
        default=all_countries
    )

    # Filter data based on selections
    df_bonds_filtered = df_bonds[
        (df_bonds['Year'] >= selected_years[0]) &
        (df_bonds['Year'] <= selected_years[1]) &
        (df_bonds['Country'].isin(selected_countries))
    ]
else:
    st.sidebar.warning("Data for filters could not be loaded.")
    df_bonds_filtered = pd.DataFrame() # Create empty dataframe to avoid errors

# --- OBJECTIVE 1: GLOBAL GREEN FINANCE MARKET ---
st.header("Objective 1: The Global Green Finance Market")

if not df_bonds_filtered.empty:
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
        fig_growth = px.line(growth_over_time, x='Year', y='Amount (USD)', markers=True,
                             labels={"Amount (USD)": "Capital Mobilised (USD)"})
        fig_growth.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.0f')
        st.plotly_chart(fig_growth, use_container_width=True)

    with c2:
        st.subheader("Issuance by Sector")
        sector_dist = df_bonds_filtered['Sector'].value_counts().nlargest(10).reset_index()
        sector_dist.columns = ['Sector', 'Count']
        fig_sector = px.pie(sector_dist, names='Sector', values='Count', hole=0.4,
                             title="Top 10 Sectors by Number of Bonds")
        st.plotly_chart(fig_sector, use_container_width=True)

    st.subheader("Top Countries by Issuance")
    top_countries = df_bonds_filtered.groupby('Country')['Amount (USD)'].sum().nlargest(15).reset_index()
    fig_countries = px.bar(top_countries, x='Country', y='Amount (USD)',
                           labels={"Amount (USD)": "Total Capital Mobilised (USD)"})
    fig_countries.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.0f')
    st.plotly_chart(fig_countries, use_container_width=True)

else:
    st.warning("Bond data could not be loaded or is empty after filtering.")

# --- OBJECTIVE 2: INVESTOR PSYCHOLOGY LANDSCAPE ---
st.header("Objective 2: The Investor Psychology Landscape")

if df_bias is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The Awareness-Bias Relationship")
        st.markdown("A strong negative correlation exists between ESG awareness and investor bias.")
        fig_scatter = px.scatter(df_bias, x='ESGAwareness', y='BiasPrevalence',
                                 hover_name='Region', trendline="ols",
                                 labels={"ESGAwareness": "ESG Awareness (%)", "BiasPrevalence": "Bias Prevalence (%)"})
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        st.subheader("Bias Prevalence by Region")
        st.markdown("Highlights regions that may require targeted ESG education.")
        bias_by_region = df_bias.sort_values('BiasPrevalence', ascending=False)
        fig_bias_bar = px.bar(bias_by_region, x='Region', y='BiasPrevalence',
                              labels={"BiasPrevalence": "Bias Prevalence (%)"})
        st.plotly_chart(fig_bias_bar, use_container_width=True)

else:
    st.warning("Behavioral bias data could not be loaded.")

# --- OBJECTIVE 3: GLOBAL BIODIVERSITY POLICY TOOLKIT ---
st.header("Objective 3: The Global Biodiversity Policy Toolkit")

if df_policy is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Policy Mix")
        st.markdown("The current toolkit is heavily dominated by subsidies and taxes.")
        policy_mix = df_policy['InstrumentType'].value_counts().reset_index()
        policy_mix.columns = ['InstrumentType', 'Count']
        fig_pie = px.pie(policy_mix, names='InstrumentType', values='Count', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("OECD vs. Non-OECD Policy Toolkits")
        st.markdown("Developed nations use a more diverse set of sophisticated tools.")
        oecd_mix = df_policy.groupby(['OECD_Status', 'InstrumentType']).size().reset_index(name='Count')
        fig_stacked = px.bar(oecd_mix, x='OECD_Status', y='Count', color='InstrumentType',
                             title="Composition of Policy Toolkits", barmode='stack')
        st.plotly_chart(fig_stacked, use_container_width=True)

else:
    st.warning("Policy data could not be loaded.")

