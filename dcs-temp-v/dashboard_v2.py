import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import sys
import os

# Import Data Engine locally
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_engine import DataEngine

st.set_page_config(page_title="V2 Customer Intelligence", layout="wide")

# --- Session State Management ---
if 'engine' not in st.session_state:
    st.session_state.engine = DataEngine()
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Config")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()
    st.subheader("Data Source")
    
    # Auto-load Logic
    engine = st.session_state.engine
    has_local, t_local, d_local, a_local = engine.load_local_files()
    
    if has_local:
        st.success("✅ Local CSV files detected!")
        if st.button("Load Local Data"):
            with st.spinner("Processing..."):
                ok, msg = engine.process_data(t_local, d_local, a_local)
                if ok:
                    engine.generate_clusters()
                    st.session_state.data_ready = True
                    st.success("Loaded & Clustered!")
                else:
                    st.error(msg)
    else:
        st.info("No local files found. Please upload.")
        t_file = st.file_uploader("Transactions", type='csv')
        d_file = st.file_uploader("Demographics", type='csv')
        a_file = st.file_uploader("Address", type='csv')
        
        if st.button("Process Uploads"):
            if t_file and d_file and a_file:
                t = pd.read_csv(t_file)
                d = pd.read_csv(d_file)
                a = pd.read_csv(a_file)
                engine.process_data(t, d, a)
                engine.generate_clusters()
                st.session_state.data_ready = True
                st.rerun()

# --- Main Dashboard ---
st.title("🤖 Dynamic Segmentation V2")

if st.session_state.data_ready:
    clusters = st.session_state.engine.clusters_df
    
    if clusters is not None and not clusters.empty:
        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Segments", "🔮 Forecast", "💬 AI Agent"])
        
        # Tab 1: Visuals
        with tab1:
            colA, colB = st.columns([2,1])
            with colA:
                st.subheader("Monthly Spending Patterns")
                # Group by cluster and mean
                trend = clusters.groupby('cluster').mean().T
                fig = px.line(trend, title="Behavioral Trajectory (Normalized)")
                st.plotly_chart(fig, use_container_width=True)
            with colB:
                st.subheader("Distribution")
                counts = clusters['cluster'].value_counts().reset_index()
                counts.columns = ['Cluster', 'Count']
                fig_pie = px.pie(counts, values='Count', names='Cluster', hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)

        # Tab 2: Prophet
        with tab2:
            st.subheader("Revenue Forecasting")
            c_id = st.selectbox("Select Segment", sorted(clusters['cluster'].unique()))
            if st.button("Run Forecast"):
                with st.spinner("Calculating..."):
                    try:
                        fc = st.session_state.engine.run_forecast(c_id)
                        fig_fc = px.line(fc, x='ds', y='yhat', title=f"Predicted Spend: Segment {c_id}")
                        st.plotly_chart(fig_fc, use_container_width=True)
                    except Exception as e:
                        st.error(f"Forecast Error (Is Prophet installed?): {e}")

        # Tab 3: AI Chat
        with tab3:
            st.subheader("Consult your Data")
            for msg in st.session_state.chat_log:
                with st.chat_message(msg['role']):
                    st.write(msg['content'])
            
            if user_input := st.chat_input("Ask me about the segments..."):
                if not api_key:
                    st.error("Please enter API Key in sidebar first.")
                else:
                    st.session_state.chat_log.append({"role": "user", "content": user_input})
                    with st.chat_message("user"):
                        st.write(user_input)
                    
                    with st.spinner("AI analyzing..."):
                        context = st.session_state.engine.get_summary_context()
                        
                        # API Call to v2_api.py
                        try:
                            res = requests.post(
                                "http://localhost:8000/chat",
                                json={"api_key": api_key, "context": context, "question": user_input}
                            )
                            if res.status_code == 200:
                                reply = res.json()["response"]
                            else:
                                reply = f"API Error {res.status_code}: {res.text}"
                        except Exception as e:
                            reply = f"Connection Failed: Is 'uvicorn' running? ({e})"
                        
                        st.session_state.chat_log.append({"role": "assistant", "content": reply})
                        with st.chat_message("assistant"):
                            st.write(reply)

    else:
        st.warning("Data loaded, but clustering returned empty result. Check logs.")
else:
    st.info("👈 Please load your data in the sidebar to begin.")