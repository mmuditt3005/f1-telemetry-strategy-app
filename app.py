import os
import boto3
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if not os.path.exists("f1_cache"):
    os.makedirs("f1_cache")
model = joblib.load('lap_time_model.pkl')
driver_enc = joblib.load('Driver_encoder.pkl')
team_enc = joblib.load('Team_encoder.pkl')
compound_enc = joblib.load('Compound_encoder.pkl')
strategy_model = joblib.load('lap_model_nosectors.pkl')
strategy_driver_enc = joblib.load('Driver_encoder_nosectors.pkl')
strategy_team_enc = joblib.load('Team_encoder_nosectors.pkl')
strategy_compound_enc = joblib.load('Compound_encoder_nosectors.pkl')

from f1_analysis import compare_drivers
from flask import Flask, Response, request, render_template_string, url_for
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def stream_video(bucket_name, file_key):
    s3_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    def generate():
        for chunk in iter(lambda: s3_object['Body'].read(1024 * 1024), b""):
            yield chunk
    return generate()

app = Flask(__name__)

@app.route("/")
def index():
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.mp4')]
    except Exception as e:
        logger.error(f"Error listing S3 files: {e}")
        files = []
    html = """
    <h1>F1 Video Stream Highlights</h1>
    <ul>
        {% for file in files %}
        <li><a href="{{ url_for('video_stream', video_key=file) }}">{{ file }}</a></li>
        {% endfor %}
    </ul>
    """
    html = """
    <h1>📺 F1 S3 Video Highlights</h1>
    <ul>
        {% for file in files %}
        <li><a href="{{ url_for('video_stream', video_key=file) }}">{{ file }}</a></li>
        {% endfor %}
    </ul>

    <hr>

    <h2>📊 Compare Telemetry (from FastF1)</h2>
    <form action="/f1-dashboard/2025/China/HAM-vs-VER" method="get">
        <button type="submit">View HAM vs VER – China 2025</button>
    </form>

    <hr>

    <h2>🧠 Try Lap Time Prediction</h2>
    <a href="/predict-lap">→ Predict a Single Lap</a>

    <hr>

    <h2>🏎️ Explore Strategy Simulation</h2>
    <a href="/strategy-dashboard">→ Generate Strategy Comparison</a>
    """

    return render_template_string(html, files=files)


@app.route("/f1-dashboard/<int:year>/<race>/<driver1>-vs-<driver2>")
def f1_dashboard(year, race, driver1, driver2):
    chart_path = compare_drivers(year, race, 'Q', driver1.upper(), driver2.upper())

    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
    videos = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.mp4')]
    video_key = videos[0] if videos else None

    if video_key:
        video_url = f"/video/{video_key}"
        video_html = f'<video width="100%" controls><source src="{video_url}" type="video/mp4"></video>'
    else:
        video_html = "<p>No video available</p>"

    return f'''
    <h1>{driver1.upper()} vs {driver2.upper()} - {race.title()} {year}</h1>
    {video_html}
    <br>
    <img src="/{chart_path}" alt="Telemetry Chart" width="100%">
    <p><a href="/">← Back to video list</a></p>
    '''
@app.route("/predict-lap", methods=["GET", "POST"])
def predict_lap():
    drivers = list(driver_enc.classes_)
    teams = list(team_enc.classes_)
    compounds = list(compound_enc.classes_)

    result = ""  # initialize result early

    html = f"""
    <h1>🏁 Predict F1 Lap Time</h1>
    <form method="POST">
        Driver: <select name="driver">
            {''.join([f'<option value="{d}">{d}</option>' for d in drivers])}
        </select><br><br>

        Team: <select name="team">
            {''.join([f'<option value="{t}">{t}</option>' for t in teams])}
        </select><br><br>

        Compound: <select name="compound">
            {''.join([f'<option value="{c}">{c}</option>' for c in compounds])}
        </select><br><br>

        Lap Number: <input name="lap_number" type="number" value="1" required><br><br>
        Sector 1 Time (s): <input name="sector1" type="text" required><br><br>
        Sector 2 Time (s): <input name="sector2" type="text" required><br><br>
        Sector 3 Time (s): <input name="sector3" type="text" required><br><br>
        <input type="submit" value="Predict">
    </form>
    """

    if request.method == "POST":
        try:
            driver = request.form["driver"]
            team = request.form["team"]
            compound = request.form["compound"]
            lap_number = int(request.form["lap_number"])
            s1 = float(request.form["sector1"])
            s2 = float(request.form["sector2"])
            s3 = float(request.form["sector3"])

            driver_code = driver_enc.transform([driver])[0]
            team_code = team_enc.transform([team])[0]
            compound_code = compound_enc.transform([compound])[0]

            X = pd.DataFrame([{
                'Driver': driver_code,
                'Team': team_code,
                'Compound': compound_code,
                'LapNumber': lap_number,
                'Sector1Time': s1,
                'Sector2Time': s2,
                'Sector3Time': s3
            }])

            pred = model.predict(X)[0]
            result = f"<h2>🎯 Predicted Lap Time: {pred:.3f} seconds</h2><br>"

        except ValueError:
            result = "<h3 style='color:red;'>⚠️ Please fill in all fields with valid numbers!</h3>"

    return render_template_string(html + result)

@app.route("/predict-strategy", methods=["GET", "POST"])
def predict_strategy():
    drivers = list(strategy_driver_enc.classes_)
    teams = list(strategy_team_enc.classes_)
    compounds = list(strategy_compound_enc.classes_)
    result = ""

    html = f"""
    <h1>🧠 Predict F1 Lap Time (Strategy Model)</h1>
    <form method="POST">
        Driver: <select name="driver">
            {''.join([f'<option value="{d}">{d}</option>' for d in drivers])}
        </select><br><br>

        Team: <select name="team">
            {''.join([f'<option value="{t}">{t}</option>' for t in teams])}
        </select><br><br>

        Compound: <select name="compound">
            {''.join([f'<option value="{c}">{c}</option>' for c in compounds])}
        </select><br><br>

        Lap Number: <input name="lap_number" type="number" value="1" required><br><br>
        <input type="submit" value="Predict Lap Time">
    </form>
    """

    if request.method == "POST":
        try:
            driver = request.form["driver"]
            team = request.form["team"]
            compound = request.form["compound"]
            lap_number = int(request.form["lap_number"])

            # Encode
            d = strategy_driver_enc.transform([driver])[0]
            t = strategy_team_enc.transform([team])[0]
            c = strategy_compound_enc.transform([compound])[0]

            X = pd.DataFrame([{
                "Driver_enc": d,
                "Team_enc": t,
                "Compound_enc": c,
                "LapNumber": lap_number
            }])

            pred = strategy_model.predict(X)[0]
            result = f"<h2>🎯 Predicted Lap Time: {pred:.3f} sec</h2>"

        except Exception as e:
            result = f"<h3 style='color:red;'>Error: {str(e)}</h3>"

    return render_template_string(html + result)

@app.route("/strategy-dashboard", methods=["GET", "POST"])
def strategy_dashboard():
    drivers = list(strategy_driver_enc.classes_)
    teams = list(strategy_team_enc.classes_)
    compounds = list(strategy_compound_enc.classes_)
    result = ""

    if request.method == "POST":
        d1 = request.form["driver1"]
        d2 = request.form["driver2"]
        t1 = request.form["team1"]
        t2 = request.form["team2"]
        compound = request.form["compound"]
        laps = int(request.form["laps"])

        lap_range = list(range(1, laps + 1))

        times_d1 = []
        times_d2 = []

        for lap in lap_range:
            d1_enc = strategy_driver_enc.transform([d1])[0]
            t1_enc = strategy_team_enc.transform([t1])[0]
            c_enc = strategy_compound_enc.transform([compound])[0]

            d2_enc = strategy_driver_enc.transform([d2])[0]
            t2_enc = strategy_team_enc.transform([t2])[0]

            X1 = pd.DataFrame([{
                "Driver_enc": d1_enc, "Team_enc": t1_enc,
                "Compound_enc": c_enc, "LapNumber": lap
            }])
            X2 = pd.DataFrame([{
                "Driver_enc": d2_enc, "Team_enc": t2_enc,
                "Compound_enc": c_enc, "LapNumber": lap
            }])

            times_d1.append(strategy_model.predict(X1)[0])
            times_d2.append(strategy_model.predict(X2)[0])

        plt.figure(figsize=(10, 5))
        plt.plot(lap_range, times_d1, label=f"{d1} ({compound})", color='blue')
        plt.title(f"{d1} Lap Time over {laps} laps on {compound}")
        plt.xlabel("Lap")
        plt.ylabel("Lap Time (s)")
        plt.grid(True)
        plt.legend()
        os.makedirs("static", exist_ok=True)
        plot1_path = f"static/{d1}_compound_strategy.png"
        plt.savefig(plot1_path)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(lap_range, times_d1, label=f"{d1}", color='blue')
        plt.plot(lap_range, times_d2, label=f"{d2}", color='red')
        plt.title(f"{d1} vs {d2} Lap Time Comparison on {compound}")
        plt.xlabel("Lap")
        plt.ylabel("Lap Time (s)")
        plt.grid(True)
        plt.legend()
        plot2_path = f"static/{d1}_vs_{d2}_comparison.png"
        plt.savefig(plot2_path)
        plt.close()

        df = pd.DataFrame({
            "Lap": lap_range,
            f"{d1}_Time": times_d1,
            f"{d2}_Time": times_d2
        })
        csv_path = f"static/{d1}_vs_{d2}_strategy.csv"
        df.to_csv(csv_path, index=False)

        result = f"""
        <h2>📈 {d1} Strategy Chart</h2>
        <img src="/{plot1_path}" width="700"><br><br>

        <h2>🆚 Comparison Chart: {d1} vs {d2}</h2>
        <img src="/{plot2_path}" width="700"><br><br>

        <h2>📄 Lap Simulation Table</h2>
        {df.to_html(index=False)}

        <br><a href='/{csv_path}' download>⬇️ Download CSV</a>
        """

    html = f"""
    <h1>🏎️ F1 Strategy Dashboard</h1>
    <form method="POST">
        Driver 1: <select name="driver1">
            {''.join([f'<option value="{d}">{d}</option>' for d in drivers])}
        </select><br><br>
        Team 1: <select name="team1">
            {''.join([f'<option value="{t}">{t}</option>' for t in teams])}
        </select><br><br>

        Driver 2: <select name="driver2">
            {''.join([f'<option value="{d}">{d}</option>' for d in drivers])}
        </select><br><br>
        Team 2: <select name="team2">
            {''.join([f'<option value="{t}">{t}</option>' for t in teams])}
        </select><br><br>

        Tire Compound: <select name="compound">
            {''.join([f'<option value="{c}">{c}</option>' for c in compounds])}
        </select><br><br>

        Number of Laps (10–25): <input type="number" name="laps" min="10" max="25" value="15"><br><br>
        <input type="submit" value="Generate Strategy">
    </form><br>
    """

    return render_template_string(html + result)


@app.route("/video/<video_key>")
def video_stream(video_key):
    return Response(stream_video(S3_BUCKET_NAME, video_key), content_type="video/mp4")

if __name__ == "__main__":
    app.run(debug=True)


