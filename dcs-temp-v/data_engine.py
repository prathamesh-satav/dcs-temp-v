import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Robust Imports ---
# Allows the app to run even if heavy ML libraries fail to install
try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMinMax
    HAS_TSLEARN = True
except ImportError:
    HAS_TSLEARN = False
    logger.warning("tslearn not found. Clustering will be disabled.")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    logger.warning("prophet not found. Forecasting will be disabled.")

class DataEngine:
    def __init__(self):
        self.master_df = None
        self.clusters_df = None

    def load_local_files(self):
        """
        Attempts to load the specific 'Cleaned' CSVs if they exist in the folder.
        """
        required = {
            'trans': 'Transactions_Cleaned.csv',
            'demo': 'CustomerDemographic_Cleaned.csv',
            'addr': 'CustomerAddress_Cleaned.csv'
        }
        
        # Check if all exist
        if all(os.path.exists(f) for f in required.values()):
            try:
                t = pd.read_csv(required['trans'])
                d = pd.read_csv(required['demo'])
                a = pd.read_csv(required['addr'])
                return True, t, d, a
            except Exception as e:
                return False, None, None, str(e)
        return False, None, None, "Files not found locally."

    def process_data(self, trans_df, demo_df, addr_df):
        """
        Merges the three dataframes into one Master Table.
        """
        try:
            # Merge 1: Transactions + Demographics
            master = pd.merge(trans_df, demo_df, on='customer_id', how='left')
            # Merge 2: + Address
            master = pd.merge(master, addr_df, on='customer_id', how='left')

            # Fix Dates
            if 'transaction_date' in master.columns:
                master['transaction_date'] = pd.to_datetime(master['transaction_date'])
            
            self.master_df = master
            return True, "Data Merged Successfully"
        except Exception as e:
            return False, f"Merge Error: {str(e)}"

    def generate_clusters(self, n_clusters=4):
        """
        1. Resamples spending to Monthly level.
        2. Normalizes data.
        3. Runs TimeSeries K-Means.
        """
        if self.master_df is None:
            raise ValueError("No data loaded.")
        
        if not HAS_TSLEARN:
            return pd.DataFrame() # Return empty if library missing

        # Prepare Time Series: Rows=Customers, Cols=Months
        df = self.master_df[['customer_id', 'transaction_date', 'list_price']].copy()
        df.set_index('transaction_date', inplace=True)
        
        # Resample Monthly Spend
        monthly = df.groupby('customer_id')['list_price'].resample('M').sum().fillna(0).reset_index()
        
        # Pivot
        pivot_df = monthly.pivot(index='customer_id', columns='transaction_date', values='list_price').fillna(0)
        
        # Scale (Normalize 0-1)
        scaler = TimeSeriesScalerMinMax()
        X_scaled = scaler.fit_transform(pivot_df.values.reshape(pivot_df.shape[0], pivot_df.shape[1], 1))
        
        # K-Means Clustering
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=42)
        labels = km.fit_predict(X_scaled)
        
        # Attach labels to data
        pivot_df['cluster'] = labels
        self.clusters_df = pivot_df
        return pivot_df

    def get_summary_context(self):
        """
        Generates a text summary of the segments for the AI Agent.
        """
        if self.clusters_df is None or self.clusters_df.empty:
            return "No clusters available."
            
        summary = "Analysis of Customer Segments:\n"
        for c in sorted(self.clusters_df['cluster'].unique()):
            cluster_data = self.clusters_df[self.clusters_df['cluster'] == c]
            avg_spend = cluster_data.mean(axis=1).mean()
            count = len(cluster_data)
            
            # Simple heuristic labeling for the AI context
            label = "Standard"
            if avg_spend > 1000: label = "High Value / VIP"
            elif avg_spend < 200: label = "Low Value / Occasional"
            
            summary += f"- Segment {c}: {count} customers. Avg Monthly Spend: ${avg_spend:.2f} ({label}).\n"
        return summary

    def run_forecast(self, cluster_id):
        """
        Runs Prophet forecast for a specific cluster.
        """
        if not HAS_PROPHET:
            raise ImportError("Prophet not installed.")
            
        # Get customers in this cluster
        cluster_customers = self.clusters_df[self.clusters_df['cluster'] == cluster_id].index
        
        # Filter master data
        df = self.master_df[['customer_id', 'transaction_date', 'list_price']].copy()
        mask = df['customer_id'].isin(cluster_customers)
        filtered = df[mask]
        
        # Aggregate Daily for Prophet
        daily = filtered.groupby('transaction_date')['list_price'].sum().reset_index()
        daily.columns = ['ds', 'y']
        
        # Fit Model
        m = Prophet()
        m.fit(daily)
        
        # Predict 6 months out
        future = m.make_future_dataframe(periods=180)
        forecast = m.predict(future)
        return forecast