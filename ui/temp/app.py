from flask import Flask, render_template, request
from utils.data_utils import fetch_stock_data
from utils.lppls import analyze_stock_data
from utils.tda import perform_tda
from utils.chart_utils import create_chart, add_lppls_to_chart, add_tda_to_chart

app = Flask(__name__)

# Global variable to store fetched data
global_data = {"data": None, "symbol": None}


@app.route("/", methods=["GET", "POST"])
def index():
    chart_data = None
    error = None

    # Retain parameters across sessions
    last_used_params = {
        "start_date": request.form.get("start_date"),
        "end_date": request.form.get("end_date"),
        "scaling_factor": float(request.form.get("scaling_factor", 0.15)),
        "min_length": int(request.form.get("min_length", 10)),
    }

    if request.method == "POST":
        action = request.form.get("action")
        if action == "fetch":
            symbol = request.form.get("symbol").upper()
            chart_data, error = fetch_stock_data(symbol, global_data)

            if global_data["data"] is not None:
                last_used_params["start_date"] = last_used_params["start_date"] or global_data["data"].index.min().strftime('%Y-%m-%d')
                last_used_params["end_date"] = last_used_params["end_date"] or global_data["data"].index.max().strftime('%Y-%m-%d')
        elif action == "lppls":
            start_date = request.form.get("start_date")
            end_date = request.form.get("end_date")
            scaling_factor = float(request.form.get("scaling_factor", 0.15))
            min_length = int(request.form.get("min_length", 10))

            last_used_params.update({
                "start_date": start_date,
                "end_date": end_date,
                "scaling_factor": scaling_factor,
                "min_length": min_length
            })

            if global_data["data"] is not None:
                chart_data, analysis_error = analyze_stock_data(
                    global_data, start_date, end_date, scaling_factor, min_length
                )
                if analysis_error:
                    error = analysis_error
                    chart_data = create_chart(global_data["data"], global_data["symbol"])
            else:
                error = "Please fetch data first."
        elif action == "tda":
            # segment_choice = int(request.form.get("segment_choice"))
            w, d, N = int(request.form.get("w")), int(request.form.get("d")), int(request.form.get("N"))

            if global_data["data"] is not None:
                try:
                    norms = perform_tda(global_data["data"]['Close'].values, w, d, N)
                    chart_data = add_tda_to_chart(global_data["data"], global_data["symbol"], norms)
                except Exception as e:
                    error = f"TDA Analysis Failed: {e}"
            else:
                error = "Please fetch data first."

    return render_template(
        'index.html',
        symbol=global_data["symbol"],
        chart_data=chart_data,
        error=error,
        last_params=last_used_params,
    )


if __name__ == "__main__":
    app.run(debug=True)

# todo: scrolling left panel
# todo: gracefully handle tda error
# todo: add volume
# todo: rescale vertically when zooming in
# todo: hourly, daily, minute
# todo: year takes only 4 digits
# todo: factor analysis tools, and make works with the script (move from deep wave)
# todo: if parameters out of range or error show message and keep existing image
# todo: tab for tda, lca, nsde
# todo: add colors for upper movemenet sector and down
# todo: remove weekend
