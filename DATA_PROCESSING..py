import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath: str):
    """
    Load the shipment dataset from CSV and preprocess:
      - parse datetimes
      - compute delivery time in minutes
      - label-encode categorical columns
    Returns:
      dataset (pd.DataFrame), label_encoders (dict)
    """
    # Load dataset
    dataset = pd.read_csv(filepath)

    # Convert datetime columns safely
    for col in ['Creation Datetime', 'Delivery Success Datetime']:
        if col in dataset.columns:
            dataset[col] = pd.to_datetime(dataset[col], errors='coerce')

    # Compute delivery time feature
    if 'Creation Datetime' in dataset.columns and 'Delivery Success Datetime' in dataset.columns:
        dataset['delivery_time_mins'] = (
            (dataset['Delivery Success Datetime'] - dataset['Creation Datetime'])
            .dt.total_seconds() / 60
        )
    else:
        dataset['delivery_time_mins'] = None

    # Encode categorical columns safely
    label_encoders = {}
    cat_cols = ['pickup hub id', 'Delivery Success Hub ID', 'Shop Name', 'Rider Name']
    for col in cat_cols:
        if col in dataset.columns:
            le = LabelEncoder()
            dataset[col] = dataset[col].astype(str)
            dataset[col] = le.fit_transform(dataset[col])
            label_encoders[col] = le

    return dataset, label_encoders


# Direct run example with your actual path
if __name__ == "__main__":
    filepath = r"C:\Users\Nagasai\Downloads\fraud-pulse-ai-main\hackathon\hackathon\dataset.csv"
    dataset, label_encoders = load_and_preprocess(filepath)
    print(dataset.head())
