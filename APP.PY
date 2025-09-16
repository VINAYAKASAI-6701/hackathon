# APP.PY - Fully Optimized Fraud Detection with Route Deviation & Carrier Reliability

from flask import Flask, request, jsonify
import numpy as np
import datetime
import joblib
import pandas as pd
from dash import Dash, dcc, html
import plotly.express as px
from blockchain import Blockchain
import os  # <-- added

# -------------------- Flask & Blockchain Setup -------------------- #
server = Flask(__name__)
blockchain = Blockchain()

# -------------------- Load Model & Scaler -------------------- #
MODEL_PATH = "fraud_model.pkl"
SCALER_PATH = "scaler.pkl"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------- Utility Functions -------------------- #
def encode_categorical(value: str) -> int:
    return hash(str(value)) % 10000

def compute_features(payload: dict):
    try:
        latitude = float(payload.get('latitude'))
        longitude = float(payload.get('longitude'))
        delay = float(payload.get('delay_minutes', 0))
        temp = float(payload.get('temperature', 5))
    except (TypeError, ValueError):
        raise ValueError("Invalid numeric values in payload")

    door_open = int(payload.get('door_open', 0))
    carrier_verified = int(payload.get('carrier_verified', 0))
    temp_anomaly = int(temp < 2 or temp > 10)
    location_anomaly = int(abs(latitude - 17.385) > 0.5 or abs(longitude - 78.4867) > 0.5)

    categorical_features = [
        encode_categorical(payload.get(field, 'NA')) for field in [
            'granular_status', 'pickup_hub_id', 'delivery_success_hub_id',
            'from_address2', 'to_address2', 'shop_name', 'rider_name'
        ]
    ]

    features = [latitude, longitude, delay, temp, door_open,
                carrier_verified, temp_anomaly, location_anomaly] + categorical_features

    anomaly_flags = {
        "carrier_verified": carrier_verified,
        "temp_anomaly": temp_anomaly,
        "door_open_flag": door_open,
        "location_anomaly": location_anomaly,
    }

    return np.array([features], dtype=float), anomaly_flags

# -------------------- Flask API Routes -------------------- #
@server.route("/", methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to Fraud Detection & Blockchain API",
        "endpoints": {
            "POST /add_shipment_event": "Add a shipment event and check fraud",
            "GET /blockchain": "View the blockchain",
            "GET /verify_chain": "Verify blockchain integrity",
            "GET /dashboard/": "Fraud visualizations dashboard"
        }
    })

@server.route('/add_shipment_event', methods=['POST'])
def add_shipment_event():
    payload = request.json or {}
    shipment_id = payload.get('shipment_id', f"SHIP-{datetime.datetime.now().timestamp()}")
    try:
        features, anomaly_flags = compute_features(payload)
        features_scaled = scaler.transform(features)
        fraud_proba = float(model.predict_proba(features_scaled)[0, 1])
        fraud_pred = fraud_proba > 0.5
    except Exception as e:
        return jsonify({"error": f"Feature extraction/prediction failed: {e}"}), 400

    event_data = {
        "shipment_id": shipment_id,
        **payload,
        **anomaly_flags,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "fraud_probability": fraud_proba,
        "fraud_detected": fraud_pred
    }
    blockchain.add_block(event_data)

    if fraud_pred or not anomaly_flags['carrier_verified']:
        server.logger.warning(f"Fraudulent/suspicious shipment detected: {shipment_id}")

    return jsonify({
        "message": "Shipment event recorded",
        "fraud_detected": fraud_pred,
        "fraud_probability": fraud_proba,
        "event_data": event_data
    })

@server.route('/blockchain', methods=['GET'])
def get_blockchain():
    return jsonify([{
        "index": b.index,
        "timestamp": b.timestamp,
        "data": b.data,
        "hash": b.hash,
        "previous_hash": b.previous_hash
    } for b in blockchain.chain])

@server.route('/verify_chain', methods=['GET'])
def verify_chain():
    return jsonify({"chain_valid": blockchain.is_chain_valid() if hasattr(blockchain, "is_chain_valid") else True})

# -------------------- Load Dataset & Preprocess -------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of APP.PY
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
df = pd.read_csv(DATA_PATH)

# Parse datetimes
df["Creation Datetime"] = pd.to_datetime(df["Creation Datetime"], errors="coerce")
df["Delivery Success Datetime"] = pd.to_datetime(df["Delivery Success Datetime"], errors="coerce")

# Compute delivery time
df["delivery_time_mins"] = (
    (df["Delivery Success Datetime"] - df["Creation Datetime"]).dt.total_seconds() / 60
)

# -------------------- Route Deviation -------------------- #
expected_hubs = (
    df.groupby(["From Address2", "To Address2"])["Delivery Success Hub ID"]
    .agg(lambda x: x.mode()[0])
    .to_dict()
)

df["route_deviation"] = df.apply(
    lambda row: int(
        row["Delivery Success Hub ID"] != expected_hubs.get(
            (row["From Address2"], row["To Address2"]), row["Delivery Success Hub ID"]
        )
    ),
    axis=1
)

# -------------------- Fraud Calculation -------------------- #
df["delay_flag"] = (df["delivery_time_mins"] > 2880).astype(int)
if "carrier_verified" not in df.columns:
    df["carrier_verified"] = 1

df["fraud_score"] = (
    df["delay_flag"] +
    (df["carrier_verified"] == 0).astype(int) +
    df["route_deviation"]
)
df["fraud"] = (df["fraud_score"] >= 2).astype(int)

# -------------------- Carrier Reliability -------------------- #
df['avg_fraud_rate'] = df.groupby(['Shop Name','Rider Name'])['fraud'].transform('mean')
df['avg_delivery_time'] = df.groupby(['Shop Name','Rider Name'])['delivery_time_mins'].transform('mean')
max_time = df['avg_delivery_time'].max()
df['delivery_norm'] = 1 - (df['avg_delivery_time'] / max_time)
df['carrier_reliability'] = 0.5*(1 - df['avg_fraud_rate']) + 0.5*df['delivery_norm']
df['carrier_verified'] = (df['carrier_reliability'] >= 0.4).astype(int)

# -------------------- Anonymize Shop and Rider Names -------------------- #
shop_mapping = {name: f"Shop {i+1}" for i, name in enumerate(df['Shop Name'].unique())}
rider_mapping = {name: f"Rider {i+1}" for i, name in enumerate(df['Rider Name'].unique())}
df['Anon Shop'] = df['Shop Name'].map(shop_mapping)
df['Anon Rider'] = df['Rider Name'].map(rider_mapping)

# -------------------- Dash Dashboard -------------------- #
dash_app = Dash(__name__, server=server, url_base_pathname='/dashboard/')

colors = {
    "primary": "#2c3e50",
    "secondary": "#34495e",
    "accent1": "#2980b9",
    "accent2": "#e74c3c",
    "accent3": "#f39c12",
    "bg": "#ecf0f1",
    "card_shadow": "0 4px 8px rgba(0,0,0,0.05)"
}

def kpi_card(title, value, color):
    return html.Div([
        html.H4(title, style={'marginBottom': '5px', 'color': colors["secondary"]}),
        html.P(f"{value}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': color})
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px',
              'boxShadow': colors["card_shadow"], 'width': '22%', 'textAlign': 'center',
              'backgroundColor': colors["bg"], 'display': 'inline-block', 'margin': '1%'})

# -------------------- KPI Cards -------------------- #
kpis = html.Div([
    kpi_card("Total Shipments", len(df), colors["accent1"]),
    kpi_card("Fraudulent Shipments", df['fraud'].sum(), colors["accent2"]),
    kpi_card("Avg Fraud Score", round(df['fraud_score'].mean(),2), colors["accent3"]),
    kpi_card("Verified Carriers", df['carrier_verified'].sum(), colors["accent1"]),
    kpi_card("Route Deviations", df['route_deviation'].sum(), colors["accent2"])
], style={'textAlign': 'center', 'marginBottom': '40px'})

# -------------------- Figures -------------------- #
fig_fraud_hist = px.histogram(df, x="fraud", title="Fraud vs Non-Fraud Distribution", color_discrete_sequence=[colors["accent1"]])

score_counts = pd.cut(df['fraud_score'], bins=10).value_counts().reset_index()
score_counts.columns = ['score_range', 'count']
score_counts = score_counts.sort_values('score_range')
score_counts['score_range'] = score_counts['score_range'].astype(str)
fig_fraud_score_donut = px.pie(score_counts, values='count', names='score_range', hole=0.5, title="Fraud Score Distribution", color_discrete_sequence=px.colors.sequential.RdBu)

fig_shop = px.line(df.groupby('Anon Shop')['fraud'].sum().reset_index(),
                   x='Anon Shop', y='fraud', title="Fraud Count by Shop",
                   markers=True, color_discrete_sequence=[colors["accent2"]])
fig_shop.update_layout(xaxis_title=None, yaxis_title=None)

fig_rider = px.histogram(df, x="Anon Rider", color="fraud", barmode="group", title="Fraud Count by Rider", color_discrete_sequence=[colors["accent3"], colors["accent2"]])
fig_rider.update_layout(xaxis_title=None, yaxis_title=None)

fig_delivery_time = px.box(df, x="fraud", y="delivery_time_mins", title="Delivery Time vs Fraud", labels={"fraud":"Fraud Flag","delivery_time_mins":"Delivery Time (mins)"}, color_discrete_sequence=[colors["accent2"]])
fig_carrier_verified = px.bar(df.groupby("carrier_verified")["fraud"].sum().reset_index(), x="carrier_verified", y="fraud", title="Fraud by Carrier Verification", labels={"carrier_verified":"Carrier Verified","fraud":"Fraud Count"}, color_discrete_sequence=[colors["accent3"]])

# -------------------- Route Deviation Pie Chart -------------------- #
route_deviation_counts = df['route_deviation'].value_counts().reset_index()
route_deviation_counts.columns = ['Deviation', 'Count']
fig_route_deviation_pie = px.pie(
    route_deviation_counts,
    values='Count',
    names='Deviation',
    title="Overall Route Deviations",
    color='Deviation',
    color_discrete_map={0: "#2980b9", 1: "#e74c3c"}  # 0 = No Deviation, 1 = Deviation
)
fig_route_deviation_pie.update_traces(
    textinfo='percent+label'
)

# -------------------- Dash Layout -------------------- #
dash_app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={'textAlign':'center','color':colors["primary"],'marginBottom':'30px'}),
    kpis,
    html.Div([
        html.Div([dcc.Graph(figure=fig_fraud_hist)], style={'width':'48%','display':'inline-block'}),
        html.Div([dcc.Graph(figure=fig_fraud_score_donut)], style={'width':'48%','display':'inline-block', 'float':'right'})
    ], style={'marginBottom':'40px'}),
    html.Div([
        html.Div([dcc.Graph(figure=fig_shop)], style={'width':'48%','display':'inline-block'}),
        html.Div([dcc.Graph(figure=fig_rider)], style={'width':'48%','display':'inline-block','float':'right'})
    ], style={'marginBottom':'40px'}),
    html.Div([
        html.Div([dcc.Graph(figure=fig_delivery_time)], style={'width':'48%','display':'inline-block'}),
        html.Div([dcc.Graph(figure=fig_carrier_verified)], style={'width':'48%','display':'inline-block','float':'right'})
    ], style={'marginBottom':'40px'}),
    html.Div([
        html.Div([dcc.Graph(figure=fig_route_deviation_pie)], style={'width':'48%','display':'inline-block'})
    ], style={'marginBottom':'40px'})
], style={'padding':'20px','fontFamily':'Arial, sans-serif'})

# -------------------- Run Server -------------------- #
if __name__ == "__main__":
    server.run(debug=True, host="0.0.0.0", port=5000)
