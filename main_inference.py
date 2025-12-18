import pandas as pd
import evidently

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.data.make_dataset import load_data
from src.features.build_features import create_features
from src.utils.config import load_config

config = load_config()

# Load training data
train_df = load_data()
# Conduct feature engineering
processed_train_df = create_features(train_df)
# Drop traget variable from training data
processed_train_df.drop(config["target"], axis=1, inplace=True)

# Load new batch of data (test.csv or live batch)
test_df = pd.read_csv(r"data\test.csv", parse_dates=["datetime"])
# Run feature engineering on new batch of data
processed_test_df = create_features(test_df)

# Conduct column_mapping
numerical_features = config["numerical_features"]
categorical_features = config["categorical_features"]

column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

# Detect drift in testing data 
report = Report([DataDriftPreset()])
report.run(current_data = processed_test_df,
               reference_data = processed_train_df,
               column_mapping=column_mapping)

report.save_html(r"reports\data_drift_report.html")
report.save_json(r"reports\data_drift_report.json")

# Notify/Log Warning/Raise Alert if data drift is detected
result = report.as_dict()
# if result["metrics"][0]["result"]["dataset_drift"]:
#     print("Data drift is detected in new batch of data")
#     print("Check for following features where data drift is detected")
# else:
#     print("Overall data drift is not detected in new batch of data")
#     print("But If it's necessary check for following features where data drift is detected")
#     # If no columns were drift detected then also print

# Extract DataDriftTable metric safely
drift_table = None
for metric in result["metrics"]:
    if metric["metric"] == "DataDriftTable":
        drift_table = metric["result"]
        break

# Defensive check
if drift_table is None:
    raise ValueError("DataDriftTable metric not found in report")

dataset_drift = drift_table["dataset_drift"]
drifted_columns = [
    col_name
    for col_name, col_info in drift_table["drift_by_columns"].items()
    if col_info["drift_detected"]
]

# Logging / Alerts
if dataset_drift:
    print("Overall data drift IS detected in the new batch of data.")
else:
    print("Overall data drift is NOT detected in the new batch of data.")

# Always report column-level drift
if drifted_columns:
    print("Drift detected in the following features:")
    for col in drifted_columns:
        print(f" - {col}")
else:
    print("No feature-level drift detected.")

# Run model prediction

# Store predictions  
