import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

VALID_CARRIERS = {'C1', 'C2', 'C3'}

def build_features_for_training(dataset: pd.DataFrame) -> pd.DataFrame:
    latitude = dataset.get('latitude', pd.Series([17.385] * len(dataset)))
    longitude = dataset.get('longitude', pd.Series([78.4867] * len(dataset)))
    delay = dataset['delivery_time_mins']
    temp = dataset.get('temperature', pd.Series([5] * len(dataset)))
    door_open = dataset.get('door_open', pd.Series([0] * len(dataset)))
    carrier_verified = dataset['carrier_verified']

    temp_anomaly = ((temp < 2) | (temp > 10)).astype(int)
    location_anomaly = (((latitude - 17.385).abs() > 0.5) | ((longitude - 78.4867).abs() > 0.5)).astype(int)

    pickup_hub = dataset['pickup hub id']
    delivery_hub = dataset['Delivery Success Hub ID']
    shop = dataset['Shop Name']
    rider = dataset['Rider Name']
    granular_status = dataset.get('Granular Status', pd.Series([0] * len(dataset)))
    from_addr = pd.factorize(dataset.get('From Address2', pd.Series(['NA'] * len(dataset))))[0]
    to_addr = pd.factorize(dataset.get('To Address2', pd.Series(['NA'] * len(dataset))))[0]

    return pd.DataFrame({
        'latitude': latitude,
        'longitude': longitude,
        'delay': delay,
        'temp': temp,
        'door_open': door_open,
        'carrier_verified': carrier_verified,
        'temp_anomaly': temp_anomaly,
        'location_anomaly': location_anomaly,
        'pickup_hub_id': pickup_hub,
        'delivery_success_hub_id': delivery_hub,
        'shop_name': shop,
        'rider_name': rider,
        'granular_status': granular_status,
        'from_address2': from_addr,
        'to_address2': to_addr
    })

def preprocess_and_label(dataset: pd.DataFrame) -> pd.DataFrame:
    for col in ['Creation Datetime', 'Delivery Success Datetime']:
        dataset[col] = pd.to_datetime(dataset[col], errors='coerce')

    dataset['delivery_time_mins'] = (dataset['Delivery Success Datetime'] - dataset['Creation Datetime']).dt.total_seconds() / 60
    dataset['carrier_verified'] = dataset.get('Carrier ID', pd.Series([0] * len(dataset))).isin(VALID_CARRIERS).astype(int)

    dataset['Granular Status'] = pd.to_numeric(dataset['Granular Status'], errors='coerce').fillna(0)
    threshold = dataset['Granular Status'].quantile(0.95)
    dataset['granular_outlier'] = (dataset['Granular Status'] > threshold).astype(int)

    delivery_flag = (dataset['delivery_time_mins'] > 60).astype(int)
    carrier_flag = 1 - dataset['carrier_verified']
    granular_flag = dataset['granular_outlier']

    dataset['fraud_score'] = (delivery_flag + carrier_flag + granular_flag) / 3
    dataset['fraud'] = (dataset['fraud_score'] > 0.5).astype(int)

    for col in ['pickup hub id', 'Delivery Success Hub ID', 'Shop Name', 'Rider Name']:
        if col in dataset.columns:
            dataset[col] = LabelEncoder().fit_transform(dataset[col].astype(str))

    return dataset

def train_and_save_model(dataset: pd.DataFrame, target_col: str, model_path: str, scaler_path: str, output_csv: str):
    X = build_features_for_training(dataset)
    y = dataset[target_col]

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved model to {model_path}, scaler to {scaler_path}")

    X_all_scaled = scaler.transform(X_imputed)
    dataset['fraud'] = model.predict(X_all_scaled)
    dataset.to_csv(output_csv, index=False)
    print(f"✅ Saved dataset with fraud predictions to {output_csv}")


if __name__ == "__main__":
    dataset_path = r"C:\Users\Nagasai\Downloads\fraud-pulse-ai-main\hackathon\hackathon\dataset.csv"
    output_csv = r"C:\Users\Nagasai\Downloads\fraud-pulse-ai-main\hackathon\hackathon\dataset_with_fraud.csv"

    dataset = pd.read_csv(dataset_path)
    dataset = preprocess_and_label(dataset)

    train_and_save_model(
        dataset,
        target_col="fraud",
        model_path=r"C:\Users\Nagasai\Downloads\fraud-pulse-ai-main\hackathon\hackathon\fraud_model.pkl",
        scaler_path=r"C:\Users\Nagasai\Downloads\fraud-pulse-ai-main\hackathon\hackathon\scaler.pkl",
        output_csv=output_csv
    )
