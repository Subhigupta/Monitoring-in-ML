import pandas as pd
import evidently

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.features.build_features import create_features

# Load new batch of data (test.csv or live batch)
test_df = pd.read_csv(r"data\test.csv", parse_dates=["datetime"])

# Run feature engineering
processed_test_df = create_features(test_df)

# Run drift monitoring
numerical_features = ['temp', 'humidity', 'windspeed', 'year', 'day', 'weekday', 'hour']
categorical_features = ['season', 'holiday', 'workingday', 'weather']

column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

report = Report([DataDriftPreset()])
report.run(current_data = processed_test_df,
               reference_data = processed_train_df_copy,
               column_mapping=column_mapping)

report.save_html(r"reports/data_drift_report.html")
report.save_json(r"reports\data_drift_report.json")

# Notify/Log Warning/Raise Alert if data drift is detected


# Run model prediction

# Store predictions  
