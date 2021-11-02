from dashboard import app1
from dashboard import app2
from dashboard import app3
from dashboard import app4

import streamlit as st


#app.py
def run_app():
    PAGES = {
        "General Analysis": app1,
        "Specific Crypto Analysis": app2,
        "On-Chain Metrics and Clustering": app3,
        "Sector Portfolios": app4
    }
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.run_app()