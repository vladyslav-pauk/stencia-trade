from ui.components.dashboard import notifications_monitor

def monitor_tab(st):
    with st.sidebar:
        st.header("Notifications Monitor")

        st.button("Stop All Notifications")

    if "monitor_thread" in st.session_state and st.session_state.monitor_thread:
        notifications_monitor(st)
    return st