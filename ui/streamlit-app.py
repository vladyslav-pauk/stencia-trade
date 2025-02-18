import streamlit as st

from views.chart import chart_tab
from views.trader import trader_tab
from views.screener import screener_tab
from views.portfolio import portfolio_tab
from views.help import help_tab
from views.monitor import monitor_tab

st.set_page_config(layout="wide")
st.title('Dashboard')

if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Help"

def update_tab(selected):
    st.session_state.selected_tab = selected

tab_selection = st.radio(
    "Tabs", ["Chart", "Trader", "Monitor", "Screener", "Portfolio", "Help"],
    index=["Chart", "Trader", "Monitor", "Screener", "Portfolio", "Help"].index(st.session_state.selected_tab),
    horizontal=True,
    on_change=update_tab,
    args=(st.session_state.selected_tab,),
    label_visibility="collapsed"
 )

if tab_selection == "Chart":
    chart_tab(st)

if tab_selection == "Trader":
    trader_tab(st)

if tab_selection == "Monitor":
    monitor_tab(st)

if tab_selection == "Screener":
    screener_tab(st)

if tab_selection == "Portfolio":
    portfolio_tab(st)

if tab_selection == "Help":
    help_tab(st)

# fixme: error handling when yfinance data not available (and other errors)
# fixme: fill in help tab
