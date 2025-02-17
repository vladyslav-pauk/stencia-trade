import streamlit as st

from components.chart import chart_tab
from components.trader import trader_tab
from components.screener import screener_tab

st.set_page_config(layout="wide")
st.title('Dashboard')

if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Chart"

def update_tab(selected):
    st.session_state.selected_tab = selected

tab_selection = st.radio("", ["Chart", "Trader", "Screener"],
                         index=["Chart", "Trader", "Screener"].index(st.session_state.selected_tab),
                         horizontal=True,
                         on_change=update_tab,
                         args=(st.session_state.selected_tab,))

if tab_selection == "Chart":
    chart_tab(st)

if tab_selection == "Trader":
    trader_tab(st)

if tab_selection == "Screener":
    screener_tab(st)
