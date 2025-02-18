import smtplib
from email.message import EmailMessage
import threading
import pandas as pd
import time

from src.utils.data import fetch_stock_data, process_data
from src.trade.strategy import support_resistance_strategy

from ui.components.indicators import add_support_resistance_data


def monitor_trading_signals(email, indicator_settings, trade_settings, ticker, strategy, interval, stop_event):
    """Continuously fetch new data and send notifications when signals occur."""

    print(f"Monitoring started for {ticker} with strategy {strategy}...")

    while not stop_event.is_set():
        date_range = [pd.to_datetime("now") - pd.Timedelta(days=10), pd.to_datetime("now")]

        try:
            data = fetch_stock_data(ticker, date_range, interval)
        except Exception as e:
            print("Error fetching stock data:", e)
            break

        data = process_data(data)
        data = add_support_resistance_data(data, indicator_settings.get("S&R", {}))

        # Apply strategy
        trade_summary = support_resistance_strategy(
            data, strategy,
            trade_settings['entry_threshold'] / 100,
            trade_settings['stop_loss'] / 100,
            trade_settings['take_profit'] / 100
        )

        # Update session state
        last_checked = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        current_price = data["Close"].iloc[-1] if not data.empty else None
        signal_count = trade_summary.shape[0]
        last_signal = trade_summary.iloc[-1].to_dict() if not trade_summary.empty else None

        # Check for new buy/sell signals
        if not trade_summary.empty:
            last_signal = trade_summary.iloc[-1]
            if last_signal["Action"] in ["Buy", "Sell"]:
                print(f"New Signal: {last_signal['Action']} @ {last_signal['Price']}")
                send_email_notification(email, last_signal, ticker)

        time.sleep(60 if interval == "1m" else 3600 if interval == "1h" else 86400)  # 1 min / 1 hour / 1 day


def send_email_notification(email, signal, ticker):
    """Send an email alert about a new trading signal."""

    subject = f"Trading Signal Alert: {signal['Action']} {ticker}"
    body = f"""
    New trading signal detected:
    - Action: {signal['Action']}
    - Price: {signal['Price']}
    - Date: {signal['Date']}
    """

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = "paukvp@gmail.com"
    msg["To"] = email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login("your_email@gmail.com", "your_password")
            server.send_message(msg)
    except Exception as e:
        print("Error sending email:", e)


def notifications(st, email):
    """Handle monitoring process and display tracking metrics."""

    if "monitor_thread" not in st.session_state:
        st.session_state.monitor_thread = None
        st.session_state.stop_event = threading.Event()
        st.session_state.last_checked = "Not started"
        st.session_state.signal_count = 0
        st.session_state.last_signal = None
        st.session_state.current_price = None

    if st.session_state.set_notifications:
        if st.session_state.monitor_thread is None or not st.session_state.monitor_thread.is_alive():
            st.session_state.stop_event.clear()

            st.session_state.monitor_thread = threading.Thread(
                target=monitor_trading_signals,
                args=(
                    email,
                    st.session_state.indicator_settings,
                    st.session_state.trade_settings,
                    st.session_state.ticker,
                    st.session_state.strategy,
                    st.session_state.interval,
                    st.session_state.stop_event
                ),
                daemon=True
            )
            st.session_state.monitor_thread.start()
            st.success("Started monitoring trading signals.")

    return st

# fixme: monitor should run in background thread
# fixme: send email notifications