import json
import os

def indicator_settings_panel(indicator, st):
    # Display relevant settings based on the selected indicator
    if indicator == "SMA":
        st.session_state.indicator_settings[indicator]['window'] = st.slider("SMA Window Size", min_value=5,
                                                                             max_value=100,
                                                                             value=st.session_state.indicator_settings[
                                                                                 indicator].get('window', 20), step=5)
    elif indicator == "EMA":
        st.session_state.indicator_settings[indicator]['window'] = st.slider("EMA Window Size", min_value=5,
                                                                             max_value=100,
                                                                             value=st.session_state.indicator_settings[
                                                                                 indicator].get('window', 20), step=5)
    elif indicator == "TDA":
        st.session_state.indicator_settings[indicator]['dimension'] = st.slider("Dimension", min_value=1, max_value=10,
                                                                                value=
                                                                                st.session_state.indicator_settings[
                                                                                    indicator].get('dimension', 5),
                                                                                step=1)
        st.session_state.indicator_settings[indicator]['delay'] = st.slider("Delay", min_value=1, max_value=20,
                                                                            value=st.session_state.indicator_settings[
                                                                                indicator].get('delay', 5), step=1)
        st.session_state.indicator_settings[indicator]['window_size'] = st.slider("Window Size", min_value=10,
                                                                                  max_value=100, value=
                                                                                  st.session_state.indicator_settings[
                                                                                      indicator].get('window_size', 20),
                                                                                  step=5)
    elif indicator == "S&R":
        st.session_state.indicator_settings[indicator]['sup_res_range'] = st.selectbox("Interval",
                                                                                       ["1wk", "2wk", "1mo", "3mo",
                                                                                        "6mo", "1yr"])
        st.session_state.indicator_settings[indicator]['num_levels'] = st.slider("Levels", min_value=1,
                                                                                 max_value=10, value=
                                                                                 st.session_state.indicator_settings[
                                                                                     indicator].get('num_levels', 1),
                                                                                 step=1)
    else:
        st.session_state.indicator_settings[indicator]['sup_res_range'], st.session_state.indicator_settings[indicator][
            'num_levels'] = None, None  # Default values

    return st


def settings_loader(st):
    SETTINGS_FILE = "ui/config/settings.json"

    if "indicator_settings" not in st.session_state:
        st.session_state.indicator_settings = {}

    available_settings = json.load(open(SETTINGS_FILE, "r")) if os.path.exists(SETTINGS_FILE) else {}
    profile_list = list(available_settings.keys())

    col1, col2 = st.columns([3, 1])
    with col1:
        profile_name = st.text_input("Save Profile", placeholder="Type profile name...",
                                     label_visibility="collapsed")
    with col2:
        save_clicked = st.button("Save")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_profile = st.selectbox("Select Profile", ["None"] + profile_list, index=0,
                                        label_visibility="collapsed")
    with col2:
        load_clicked = st.button("Load")

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
