import onnxruntime as ort
import onnx

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

def load_models():
    
    models = {}
    
    # Load Random Forest (ONNX format)
    try:
        models["random_forest"] = ort.InferenceSession("models/rf_model.onnx")
        logger.info("Loaded Random Forest model (ONNX)")
    except Exception as e:
        logger.info(f"✗ Failed to load RF model: {e}")
    
    # Load XGBoost (pickle format)
    try:
        models["xgboost"] = ort.InferenceSession("models/xgb_model.onnx")
        logger.info("Loaded XGBoost model (ONNX)")
    except Exception as e:
        logger.info(f"Failed to load XGB model: {e}")
    
    return models

def main():

    logger.info("A/B Testing...")

    ab_config = config["ab_testing"]

    if not ab_config["enabled"]:
        logger.info("A/B testing is disabled in config.toml")
        logger.info("Set [ab_testing] enabled = true to run experiments")
        return
    
    experiments = ab_config["experiments"]
    print("Experiments",experiments)


if __name__ == "__main__":
    main()