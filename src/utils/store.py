import os
from typing import Any, Dict
import onnxmltools
import json

from src.models.classifier import Classifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from src.utils.config import load_config

# Principle: Single responsibility principle
config = load_config()
PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
TOTAL_FEATURES = len(config["features"])

class Store:

    model_dir = MODEL_DIR
    reports_dir = REPORTS_DIR
    total_features = TOTAL_FEATURES

    def put_rf_onnx(self, filepath: str, python_object: Any) -> None:
        if not python_object:
            raise TypeError("python_object must be non-zero, non-empty, and not None")
        
        # Define input type and shape
        initial_type = [('float_input', FloatTensorType([None, TOTAL_FEATURES]))]

        # Convert to ONNX
        onnx_model = convert_sklearn(python_object, initial_types=initial_type)

        # Save the model
        with open(filepath, "wb") as f:
            f.write(onnx_model.SerializeToString())
    
    def put_xgb_onnx(self, filepath: str, python_object: Any) -> None:
        if not python_object:
            raise TypeError("python_object must be non-zero, non-empty, and not None")
        
        # Define input type and shape
        initial_type = [('float_input', FloatTensorType([None, TOTAL_FEATURES]))]

        # Convert to ONNX
        onnx_model = onnxmltools.convert_xgboost(python_object, initial_types=initial_type)

        # Save the model
        with open(filepath, "wb") as f:
            f.write(onnx_model.SerializeToString())

    def put_json(self, filepath: str, dic: Dict) -> None:
        if not isinstance(dic, dict):
            raise TypeError(f"dic must be of type dict, got {type(dic)}")
        with open(filepath, "w") as f:
            json.dump(dic, f)

class AssignmentStore(Store):

    def put_rf_model(self, filepath: str, model: Classifier) -> None:
        filepath = os.path.join(self.model_dir, filepath)
        self.put_rf_onnx(filepath, model)
    
    def put_xgb_model(self, filepath: str, model: Classifier) -> None:
        filepath = os.path.join(self.model_dir, filepath)
        self.put_xgb_onnx(filepath, model)

    def put_metrics(self, filepath: str, metrics: Dict[str, float]) -> None:
        filepath = os.path.join(self.reports_dir, filepath)
        self.put_json(filepath, metrics)