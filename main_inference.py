import os
import pandas as pd
import evidently
import onnx
import onnxruntime as ort
import numpy as np
import json

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.data.make_dataset import load_data
from src.features.build_features import create_features
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

# Load training data
train_df = load_data()
# Conduct feature engineering
processed_train_df = create_features(train_df)
# Drop traget variable from training data
processed_train_df.drop(config["target"], axis=1, inplace=True)

# Load new batch of data (test.csv or live batch)
logger.info("Loading New batch of Data and Conducting Feature Engineering...")
test_df = pd.read_csv(r"data\test.csv", parse_dates=["datetime"])
# Run feature engineering on new batch of data
processed_test_df = create_features(test_df)

# Conduct column_mapping
numerical_features = config["numerical_features"]
categorical_features = config["categorical_features"]
column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

# Detect data drift in testing data 
logger.info("Monitoring Dataset Drift...")
report = Report([DataDriftPreset()])
report.run(current_data = processed_test_df,
               reference_data = processed_train_df,
               column_mapping=column_mapping)
os.makedirs("monitoring", exist_ok=True)
report.save_html(r"monitoring\data_drift_report.html")
report.save_json(r"monitoring\data_drift_report.json")
result = report.as_dict()

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

# Always report column-level drift
def feature_level_drift(drifted_columns):
    if drifted_columns:
        logger.info("Drift detected in the following features:")
        for col in drifted_columns:
            logger.info(f" - {col}")
    else:
        logger.info("No feature-level drift detected.")

# Functionality to check severness to drift
def drift_decision(dataset_drift, drifted_columns, all_columns):
    drift_ratio = len(drifted_columns) / len(all_columns)

    if not dataset_drift:
        return "PREDICT"

    if drift_ratio < 0.3:
        return "PREDICT_WITH_WARNING"

    return "BLOCK_AND_RETRAIN"

# Functionality to store predictions over the target variable
def store_predictions(onnx_output):
    test_df["count_predictions"] = onnx_output
    output_path = r"data\predictions.csv"
    test_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

# Functionality to use pre-trained model for predictions
def model_inference(model):
    if model=="xgboost":
        onnx_model = onnx.load(r"models\xgb_model.onnx")
    else:
        onnx_model = onnx.load(r"models\rf_model.onnx")

    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(r"models\xgb_model.onnx")
    input_name = session.get_inputs()[0].name

    test_input = processed_test_df[config["features"]].values.astype(np.float32)
    onnx_output = session.run(None, {input_name: test_input})
    onnx_output = onnx_output[0][:,0]

    logger.info(f"Model Inference has been conducted...")
    
    return onnx_output

# Functionality to load model_promotion.json file and load the winner model
def load_model():
    with open("evaluation/model_promotion.json", "r") as f:
        result = json.load(f)
    selected_model = result["selected_model"]

    if selected_model=="treatment":
        model = config["ab_testing"]["experiments"]["model_comparison_v1"]["variants"]["treatment"]["model"]
    else:
        model = config["ab_testing"]["experiments"]["model_comparison_v1"]["variants"]["control"]["model"]
    
    return model

# Notify/Log Warning/Raise Alert if data drift is detected
decision = drift_decision(dataset_drift, drifted_columns, drift_table["drift_by_columns"])

# Fetch the model name
model = load_model()

if decision == "PREDICT":
    logger.info("No drift detected. Proceeding with predictions...")
    feature_level_drift(drifted_columns)
    onnx_output = model_inference(model)
    store_predictions(onnx_output)

elif decision == "PREDICT_WITH_WARNING":
    logger.warning("Mild drift detected. Proceeding with predictions...")
    feature_level_drift(drifted_columns)
    onnx_output = model_inference()
    store_predictions(onnx_output)

elif decision == "BLOCK_AND_RETRAIN":
    logger.critical("Severe drift detected! Blocking predictions and triggering retraining...")
    feature_level_drift(drifted_columns)
    raise RuntimeError("Prediction blocked due to severe data drift...")