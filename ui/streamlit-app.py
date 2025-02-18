import streamlit as st

from components.chart import chart_tab
from components.trader import trader_tab
from components.screener import screener_tab
from components.portfolio import portfolio_tab
from components.help import help_tab

st.set_page_config(layout="wide")
st.title('Dashboard')

if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Help"

def update_tab(selected):
    st.session_state.selected_tab = selected

tab_selection = st.radio("", ["Chart", "Trader", "Screener", "Portfolio", "Help"],
                         index=["Chart", "Trader", "Screener", "Portfolio", "Help"].index(st.session_state.selected_tab),
                         horizontal=True,
                         on_change=update_tab,
                         args=(st.session_state.selected_tab,))

if tab_selection == "Chart":
    chart_tab(st)

if tab_selection == "Trader":
    trader_tab(st)

if tab_selection == "Screener":
    screener_tab(st)

if tab_selection == "Portfolio":
    portfolio_tab(st)

if tab_selection == "Help":
    help_tab(st)
