"""Prediction service integration for A/B testing."""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import pickle
import hashlib
import numpy as np
import onnxruntime as ort

from .data_splitter import ABTestDataSplitter


class ABTestPredictionService:
    """
    Wraps model prediction with A/B testing capabilities.
    Uses holdout set from train.csv for evaluation with ground truth.
    """
    
    def __init__(self, models: Dict[str, Any], feature_columns: List[str],ab_config: dict,
                 experiment_id: str):

        """Initialize service with all components."""
        self.ab_config = ab_config
        self.experiment_id = experiment_id
        self.models = models
        self.feature_columns = feature_columns
    
    def run_experiment(self, ab_test_data: pd.DataFrame, target_column: str = "count") -> Dict[str, Any]:
        """
        Run A/B test experiment on holdout data.
        
        Args:
            ab_test_data: Holdout dataframe with features AND ground truth
            target_column: Name of the ground truth column
            
        Returns:
            Summary dict with predictions per variant and metrics
        """
        results = {
            "total_predictions": 0,
            "control_count": 0,
            "treatment_count": 0,
            "predictions": []
        }
        
        for idx, row in ab_test_data.iterrows():
            request_id = f"req_{idx}"
            features = row[self.feature_columns].to_dict()
            ground_truth = row[target_column]
            print(request_id, features,ground_truth)
            
            prediction, metadata = self.predict_single(features=features,ground_truth=ground_truth,
                                                       request_id=request_id)
            
              
            results["total_predictions"] += 1
            if metadata["variant"] == "control":
                results["control_count"] += 1
            else:
                results["treatment_count"] += 1
            
            results["predictions"].append({
                "request_id": request_id,
                "variant": metadata["variant"],
                "model": metadata["model"],
                "prediction": prediction,
                "ground_truth": ground_truth,
                "error": prediction - ground_truth
            })
            
            break
        
        return results

    def assign_variant(self, request_id: str, experiment_id: str, experiment: dict):
        """
        Deterministically assign variant & model using hashing.
        """
        # Hash request_id + experiment_id → [0,1)
        key = f"{request_id}_{experiment_id}".encode("utf-8")
        bucket = int(hashlib.md5(key).hexdigest(), 16) % 10000 / 10000.0

        cumulative = 0.0
        for variant_name, variant_data in experiment["variants"].items():
            cumulative += variant_data["traffic_split"]
            if bucket <= cumulative:
                print(variant_name, variant_data["model"])
                return variant_name, variant_data["model"]

        # Safety fallback
        first_variant = next(iter(experiment["variants"].items()))
        return first_variant[0], first_variant[1]["model"]
    
    def predict_single(self, features: Dict[str, Any], ground_truth: float, 
                       request_id: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Make a single prediction with A/B routing.
        
        Args:
            features: Input features for prediction
            ground_truth: Actual value from holdout set
            request_id: Optional request ID (generated if not provided)
            
        Returns:
            Tuple of (prediction, metadata)
        """
        if request_id is None:
            request_id = self._generate_request_id()
        
        start_time = time.time()

        experiment = self.ab_config["experiments"][self.experiment_id]
        
        # Get variant assignment
        variant, model_name = self.assign_variant(request_id=request_id, experiment_id=self.experiment_id,
                                             experiment=experiment)
        print(variant, model_name)

        model = self.models[model_name]

        # Prepare ONNX input
        input_array = np.array([list(features.values())],dtype=np.float32)

        prediction = model.run(None, {model.get_inputs()[0].name: input_array})[0][0]

        metadata = {
            "request_id": request_id,
            "experiment_id": self.experiment_id,
            "variant": variant,
            "model": model_name
        }

        print(prediction, metadata)

        return prediction, metadata
       
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return f"req_{uuid.uuid4().hex[:12]}"
