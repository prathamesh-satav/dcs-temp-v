import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import sys
import os

# Ensure we can import backend_logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend_logic import DataEngine

# Page Config
st.set_page_config(page_title="V2 Customer Intelligence", layout="wide")

# Session State
if 'engine' not in st.session_state:
    st.session_state.engine = DataEngine()
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.header("2. Data Upload")
    t_file = st.file_uploader("Transactions", type='csv')
    d_file = st.file_uploader("Demographics", type='csv')
    a_file = st.file_uploader("Address", type='csv')
    
    if st.button("Process Data"):
        if t_file and d_file and a_file:
            with st.spinner("Processing & Clustering..."):
                # Load CSVs into Dataframes
                t_df = pd.read_csv(t_file)
                d_df = pd.read_csv(d_file)
                a_df = pd.read_csv(a_file)
                
                # Run Engine
                success, msg = st.session_state.engine.process_files(t_df, d_df, a_df)
                if success:
                    # Run clustering (safely)
                    clusters = st.session_state.engine.generate_clusters()
                    
                    if not clusters.empty:
                        st.session_state.data_processed = True
                        st.success("Segmentation Complete!")
                    else:
                        st.warning("Data Processed, but Clustering failed. (Is 'tslearn' installed?)")
                        st.session_state.data_processed = True # Allow viewing data even if clustering fails
                else:
                    st.error(f"Error: {msg}")
        else:
            st.warning("Please upload all 3 files.")

# --- Main App ---
st.title("🤖 Intelligent Customer Segmentation V2")

if st.session_state.data_processed:
    engine = st.session_state.engine
    clusters = engine.clusters_df
    
    # --- Tabs ---
    tab_dash, tab_chat = st.tabs(["📊 Interactive Dashboard", "💬 AI Analyst"])
    
    with tab_dash:
        if clusters is not None and not clusters.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Spending Trajectories")
                # Calculate average trend per cluster
                avg_trend = clusters.groupby('cluster').mean().T
                fig = px.line(avg_trend, title="Average Monthly Spend by Segment")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("Segment Size")
                counts = clusters['cluster'].value_counts().reset_index()
                counts.columns = ['Cluster', 'Count']
                fig_pie = px.pie(counts, values='Count', names='Cluster', hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            st.subheader("🔮 Predictive Forecasting")
            selected_cluster = st.selectbox("Select Cluster to Forecast", sorted(clusters['cluster'].unique()))
            
            if st.button("Generate Forecast"):
                with st.spinner("Forecasting..."):
                    try:
                        forecast = engine.forecast_cluster(selected_cluster)
                        fig_prophet = px.line(forecast, x='ds', y='yhat', title=f"6-Month Forecast for Cluster {selected_cluster}")
                        st.plotly_chart(fig_prophet, use_container_width=True)
                    except ImportError:
                        st.error("Forecasting requires 'prophet' library. Please install it.")
                    except Exception as e:
                        st.error(f"Forecast Error: {e}")
        else:
            st.warning("Clusters data is not available. Please check if 'tslearn' is installed properly.")

    with tab_chat:
        st.subheader("Ask the AI about your customers")
        
        # Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        # Chat Input
        if prompt := st.chat_input("e.g., Which segment is most at risk of churning?"):
            if not api_key:
                st.error("Please enter API Key in sidebar.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Get Answer
                with st.spinner("AI is thinking..."):
                    context = engine.get_summary_text()
                    
                    try:
                        payload = {
                            "api_key": api_key,
                            "context_summary": context,
                            "question": prompt
                        }
                        # API Call
                        res = requests.post("http://localhost:8000/analyze", json=payload)
                        if res.status_code == 200:
                            ans = res.json()["answer"]
                        else:
                            ans = f"API Error: {res.text}. (Make sure uvicorn is running)"
                    except Exception as e:
                        ans = f"Connection Error: Is the FastAPI server running? (Run 'uvicorn v2_api:app --reload' in terminal). Error: {e}"

                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    with st.chat_message("assistant"):
                        st.write(ans)

else:
    st.info("👈 Upload your files in the sidebar to begin.")