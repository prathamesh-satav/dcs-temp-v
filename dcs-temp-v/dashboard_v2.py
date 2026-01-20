import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Customer Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Modular Data Processing Functions ---

@st.cache_data
def load_data(uploaded_files=None):
    """
    Loads data from either uploaded files or local default paths.
    Returns: Tuple of (transactions, demographics, address) or None.
    """
    # Default file paths
    files = {
        'trans': 'Transactions_Cleaned.csv',
        'demo': 'CustomerDemographic_Cleaned.csv',
        'addr': 'CustomerAddress_Cleaned.csv'
    }

    try:
        # Check if using uploaded files
        if uploaded_files and all(k in uploaded_files for k in files.keys()):
            t = pd.read_csv(uploaded_files['trans'])
            d = pd.read_csv(uploaded_files['demo'])
            a = pd.read_csv(uploaded_files['addr'])
            return t, d, a
        
        # Fallback to local files
        elif all(os.path.exists(f) for f in files.values()):
            t = pd.read_csv(files['trans'])
            d = pd.read_csv(files['demo'])
            a = pd.read_csv(files['addr'])
            return t, d, a
        
        else:
            return None, None, None
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def preprocess_and_merge(trans, demo, addr):
    """
    Merges datasets and handles basic cleaning.
    """
    # Merge Logic
    master = pd.merge(trans, demo, on='customer_id', how='left')
    master = pd.merge(master, addr, on='customer_id', how='left')

    # Date Conversion
    if 'transaction_date' in master.columns:
        master['transaction_date'] = pd.to_datetime(master['transaction_date'])
    
    # Filter valid IDs
    master = master.dropna(subset=['customer_id'])
    
    return master

@st.cache_data
def calculate_lrfm_features(master_df):
    """
    Calculates LRFM metrics resampled by Month for Time Series Clustering.
    """
    # Prepare base dataframe
    df = master_df[['customer_id', 'transaction_date', 'list_price']].copy()
    df.set_index('transaction_date', inplace=True)

    # 1. Resample to Monthly Frequency
    # We aggregate 'list_price' as Monetary Value
    monthly_data = df.groupby('customer_id')['list_price'].resample('M').sum().fillna(0).reset_index()
    
    # 2. Pivot for Clustering (Rows: Customers, Cols: Time)
    # This creates the "signature" or "trajectory" for each customer
    pivot_df = monthly_data.pivot(index='customer_id', columns='transaction_date', values='list_price').fillna(0)
    
    return monthly_data, pivot_df

# --- 2. Advanced Analytics Functions (ML) ---

@st.cache_resource
def perform_clustering(pivot_df, n_clusters=4, metric="euclidean"):
    """
    Executes TimeSeries K-Means Clustering.
    """
    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.preprocessing import TimeSeriesScalerMinMax
    except ImportError:
        st.error("The 'tslearn' library is missing. Please install it to use clustering.")
        return None

    # Normalization (Crucial for shape-based clustering)
    scaler = TimeSeriesScalerMinMax()
    X_scaled = scaler.fit_transform(pivot_df.values.reshape(pivot_df.shape[0], pivot_df.shape[1], 1))

    # Clustering Model
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=42)
    labels = model.fit_predict(X_scaled)
    
    return labels

@st.cache_resource
def generate_forecast(cluster_data, periods=6):
    """
    Generates forecast using Facebook Prophet.
    """
    try:
        from prophet import Prophet
    except ImportError:
        st.error("The 'prophet' library is missing. Please install it to use forecasting.")
        return None

    # Prepare data for Prophet (ds, y)
    prophet_df = cluster_data.rename(columns={'transaction_date': 'ds', 'list_price': 'y'})
    
    # Group by date to get daily/monthly sum for the whole cluster
    prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()

    # Model Fitting
    m = Prophet()
    m.fit(prophet_df)
    
    # Prediction
    future = m.make_future_dataframe(periods=periods, freq='M')
    forecast = m.predict(future)
    
    return forecast

# --- 3. UI Rendering Components ---

def render_sidebar():
    """Renders the sidebar controls."""
    st.sidebar.title("⚙️ Configuration")
    
    # File Uploader (Optional)
    with st.sidebar.expander("📂 Data Source", expanded=True):
        st.info("Default: Loads local CSVs automatically.")
        use_upload = st.checkbox("Upload new files?")
        files = None
        if use_upload:
            t = st.file_uploader("Transactions", type='csv')
            d = st.file_uploader("Demographics", type='csv')
            a = st.file_uploader("Address", type='csv')
            if t and d and a:
                files = {'trans': t, 'demo': d, 'addr': a}
    
    # Model Parameters
    st.sidebar.divider()
    st.sidebar.subheader("🧠 Model Settings")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 4)
    metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "dtw"], index=0)
    
    return files, n_clusters, metric

def render_overview_tab(master_df):
    """Renders basic data statistics and raw data view."""
    st.header("📊 Data Overview")
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{master_df['customer_id'].nunique():,}")
    c2.metric("Total Transactions", f"{len(master_df):,}")
    c3.metric("Total Revenue", f"${master_df['list_price'].sum():,.0f}")
    c4.metric("Avg Order Value", f"${master_df['list_price'].mean():.2f}")
    
    # Charts
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Transactions over Time")
        daily_sales = master_df.groupby('transaction_date')['list_price'].sum().reset_index()
        fig = px.line(daily_sales, x='transaction_date', y='list_price', title="Daily Sales Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    with c_right:
        st.subheader("Customer Demographics")
        if 'gender' in master_df.columns:
            fig_g = px.pie(master_df, names='gender', title="Gender Distribution")
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.info("Gender data not available.")

    with st.expander("🔎 View Raw Data"):
        st.dataframe(master_df.head(100))

def render_segmentation_tab(pivot_df, monthly_data, labels):
    """Renders clustering results."""
    st.header("🧩 Dynamic Segmentation")
    st.markdown("We use **TimeSeries K-Means** to group customers based on their spending trajectory over time.")
    
    # Attach labels to pivot and melt for visualization
    pivot_viz = pivot_df.copy()
    pivot_viz['Cluster'] = labels
    
    # Calculate Cluster Averages (The "Centroids")
    cluster_means = pivot_viz.groupby('Cluster').mean().T
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Cluster Spending Trajectories")
        fig = px.line(cluster_means, title="Average Monthly Spend by Cluster (The 'Shape' of each Segment)")
        fig.update_layout(xaxis_title="Month", yaxis_title="Average Spend ($)")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Segment Size")
        counts = pivot_viz['Cluster'].value_counts().reset_index()
        counts.columns = ['Cluster', 'Count']
        fig_pie = px.pie(counts, values='Count', names='Cluster', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Detailed Cluster Analysis
    st.subheader("Segment Characteristics")
    selected_c = st.selectbox("Select Cluster to Inspect", sorted(pivot_viz['Cluster'].unique()))
    
    # Filter data for selected cluster
    cluster_customers = pivot_viz[pivot_viz['Cluster'] == selected_c].index
    subset_raw = monthly_data[monthly_data['customer_id'].isin(cluster_customers)]
    
    avg_val = subset_raw['list_price'].mean()
    st.write(f"**Cluster {selected_c} Insight:** Contains {len(cluster_customers)} customers. Average monthly transaction value is ${avg_val:.2f}.")

def render_forecast_tab(monthly_data, pivot_df, labels):
    """Renders forecasting logic."""
    st.header("🔮 Future Revenue Forecasting")
    st.markdown("We use **Facebook Prophet** to predict the future spending trend of specific segments.")
    
    # Prepare Data
    pivot_viz = pivot_df.copy()
    pivot_viz['Cluster'] = labels
    
    col_sel, col_btn = st.columns([1, 4])
    with col_sel:
        c_id = st.selectbox("Select Cluster to Forecast", sorted(pivot_viz['Cluster'].unique()))
    
    # Logic
    cluster_customers = pivot_viz[pivot_viz['Cluster'] == c_id].index
    cluster_monthly_data = monthly_data[monthly_data['customer_id'].isin(cluster_customers)]
    
    if st.button("Generate 6-Month Forecast"):
        with st.spinner(f"Training Prophet model for Cluster {c_id}..."):
            forecast = generate_forecast(cluster_monthly_data)
            
            if forecast is not None:
                # Plotting with Plotly for interactivity
                st.subheader(f"Forecast Results: Cluster {c_id}")
                
                # Main line
                fig = go.Figure()
                
                # Historical Trend (we can plot the actuals if we want, but let's focus on the forecast line)
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                
                # Uncertainty Intervals
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_upper'],
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_lower'],
                    mode='lines', line=dict(width=0), fill='tonexty',
                    fillcolor='rgba(0, 100, 80, 0.2)', name='Uncertainty Interval'
                ))
                
                fig.update_layout(title="Predicted Spending Trend", xaxis_title="Date", yaxis_title="Revenue ($)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

# --- 4. Main Application Execution ---

def main():
    # A. Sidebar & Setup
    uploaded_files, n_clusters, metric = render_sidebar()
    
    # B. Data Loading
    with st.spinner("Loading and merging data..."):
        t, d, a = load_data(uploaded_files)
        
    if t is None:
        st.warning("⚠️ Data not found. Please ensure CSV files are in the folder or upload them in the sidebar.")
        st.stop()
        
    # C. Processing
    master_df = preprocess_and_merge(t, d, a)
    monthly_data, pivot_df = calculate_lrfm_features(master_df)
    
    # D. Clustering (Run immediately to have labels ready)
    labels = perform_clustering(pivot_df, n_clusters, metric)
    
    if labels is None:
        st.stop() # Stop if clustering library missing

    # E. Main UI Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧩 Segmentation", "🔮 Forecasting"])
    
    with tab1:
        render_overview_tab(master_df)
        
    with tab2:
        render_segmentation_tab(pivot_df, monthly_data, labels)
        
    with tab3:
        render_forecast_tab(monthly_data, pivot_df, labels)

if __name__ == "__main__":
    main()