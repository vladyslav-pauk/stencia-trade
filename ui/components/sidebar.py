import os
import json


def indicator_settings_loader_pane(st):
    SETTINGS_FILE = "ui/config/indicator_settings.json"

    if "indicator_settings" not in st.session_state:
        st.session_state.indicator_settings = {ind: {} for ind in ["SMA", "EMA", "TDA", "S&R", "FIB", "TRE"]}

    available_settings = json.load(open(SETTINGS_FILE, "r")) if os.path.exists(SETTINGS_FILE) else {}
    profile_list = list(available_settings.keys())

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_profile = st.selectbox("Select Profile", ["None"] + profile_list, index=0,
                                        label_visibility="collapsed")
    with col2:
        load_clicked = st.button("Load")

    col1, col2 = st.columns([3, 1])
    with col1:
        profile_name = st.text_input("Save Profile", placeholder="Type profile name...",
                                     label_visibility="collapsed")
    with col2:
        save_clicked = st.button("Save")

    if save_clicked:
        if profile_name.strip():
            available_settings[profile_name] = {
                "indicators": st.session_state.indicators,
                "settings": st.session_state.indicator_settings
            }
            with open(SETTINGS_FILE, "w") as f:
                json.dump(available_settings, f, indent=4)
            st.success(f"Settings saved as '{profile_name}'")
        else:
            st.warning("Please enter a profile name before saving.")

    if load_clicked:
        if selected_profile != "None" and selected_profile in available_settings:
            loaded_profile = available_settings[selected_profile]

            st.session_state.indicator_settings = {}
            st.session_state.indicator_settings = loaded_profile.get("settings", {})

            st.session_state.indicators = []
            st.session_state.indicators = loaded_profile.get("indicators", [])

            st.rerun()
            st.info(f"Settings loaded from '{selected_profile}'")
        else:
            st.warning("Selected profile not found or invalid.")

    return st


def trader_settings_loader_pane(st):
    SETTINGS_FILE = "ui/config/trader_settings.json"

    if "indicator_settings" not in st.session_state:
        st.session_state.indicator_settings = {ind: {} for ind in ["SMA", "EMA", "TDA", "S&R", "FIB", "TRE"]}

    if "trade_settings" not in st.session_state:
        st.session_state.trade_settings = {}

    if "indicators" not in st.session_state:
        st.session_state.indicators = []

    available_settings = json.load(open(SETTINGS_FILE, "r")) if os.path.exists(SETTINGS_FILE) else {}
    profile_list = list(available_settings.keys())

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_profile = st.selectbox("Select Profile", ["None"] + profile_list, index=0,
                                        label_visibility="collapsed")
    with col2:
        load_clicked = st.button("Load")

    col1, col2 = st.columns([3, 1])
    with col1:
        profile_name = st.text_input("Save Profile", placeholder="Type profile name...",
                                     label_visibility="collapsed")
    with col2:
        save_clicked = st.button("Save")

    if save_clicked:
        if profile_name.strip():
            available_settings[profile_name] = {
                "indicators": st.session_state.indicators,
                "settings": st.session_state.indicator_settings,
                "trade_settings": st.session_state.trade_settings
            }
            with open(SETTINGS_FILE, "w") as f:
                json.dump(available_settings, f, indent=4)
            st.success(f"Settings saved as '{profile_name}'")
        else:
            st.warning("Please enter a profile name before saving.")

    if load_clicked:
        if selected_profile != "None" and selected_profile in available_settings:
            loaded_profile = available_settings[selected_profile]

            st.session_state.indicator_settings = {}
            st.session_state.indicator_settings = loaded_profile.get("settings", {})

            st.session_state.indicators = []
            st.session_state.indicators = loaded_profile.get("indicators", [])

            st.session_state.trade_settings = {}
            st.session_state.trade_settings = loaded_profile.get("trade_settings", {})

            st.rerun()
            st.info(f"Settings loaded from '{selected_profile}'")
            # todo: info load not showing both in trader and charter
        else:
            st.warning("Selected profile not found or invalid.")

    return st


def indicator_settings_pane(indicator, st):

    if indicator == "SMA":
        st.session_state.indicator_settings[indicator]['window'] = st.slider(
            "SMA Window Size",
            min_value=5,
            max_value=100,
            value=st.session_state.indicator_settings[indicator].get('window', 20),
            step=5
        )

    elif indicator == "EMA":
        st.session_state.indicator_settings[indicator]['window'] = st.slider(
            "EMA Window Size",
            min_value=5,
            max_value=100,
            value=st.session_state.indicator_settings[indicator].get('window', 20),
            step=5
        )

    elif indicator == "TDA":
        st.session_state.indicator_settings[indicator]['dimension'] = st.slider(
            "Dimension",
            min_value=1,
            max_value=10,
            value=
            st.session_state.indicator_settings[indicator].get('dimension', 5),
            step=1
        )
        st.session_state.indicator_settings[indicator]['delay'] = st.slider(
            "Delay",
            min_value=1,
            max_value=20,
            value=st.session_state.indicator_settings[indicator].get('delay', 5),
            step=1
        )
        st.session_state.indicator_settings[indicator]['window_size'] = st.slider(
            "Window Size",
            min_value=10,
            max_value=100,
            value=st.session_state.indicator_settings[indicator].get('window_size', 20),
            step=5
        )

    elif indicator == "S&R":
        sup_res_options = ["1wk", "2wk", "1mo", "3mo", "6mo", "1yr"]
        default_value = st.session_state.indicator_settings[indicator].get('sup_res_range', "1mo")

        # Ensure the default value is in the list and get its index
        default_index = sup_res_options.index(
            default_value) if default_value in sup_res_options else 2  # "1mo" is at index 2

        st.session_state.indicator_settings[indicator]['sup_res_range'] = st.selectbox(
            "Interval",
            sup_res_options,
            index=default_index
        )
        st.session_state.indicator_settings[indicator]['num_levels'] = st.slider(
            "Levels",
            min_value=1,
            max_value=10,
            value=st.session_state.indicator_settings[indicator].get('num_levels', 1),
            step=1
        )

    else:
        st.session_state.indicator_settings[indicator]['sup_res_range'], st.session_state.indicator_settings[indicator][
            'num_levels'] = None, None

    return st


def trader_settings_pane(st):
    st.session_state.trade_settings['entry_threshold'] = st.slider(
        "Entry Threshold (%)", 0.0, 1.0, st.session_state.trade_settings.get('entry_threshold', 0.01), step=0.01
    )
    st.session_state.trade_settings['stop_loss'] = st.slider(
        "Stop Loss (%)", 0.01, 10.0, st.session_state.trade_settings.get('stop_loss', 3.0), step=0.1
    )
    st.session_state.trade_settings['take_profit'] = st.slider(
        "Take Profit (%)", 0.01, 10.0, st.session_state.trade_settings.get('take_profit', 5.0), step=0.1
    )

    st.session_state.trade_settings['best_parameters'] = st.button("Optimize Parameters")
    return st


def info_pane(st):
    st.sidebar.subheader('About')
    st.sidebar.info(
        'This dashboard provides stock data and various indicators. Use the sidebar to customize your view.'
    )
    return st
