import onnxruntime as ort
from datetime import datetime
import os, json

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.make_dataset import load_data
from src.features.build_features import create_features
from src.ab_testing.data_splitter import ABTestDataSplitter
from src.ab_testing.service import ABTestPredictionService
from src.ab_testing.metrics import calculate_ab_metrics, compare_variants, save_metrics
from src.ab_testing.analyzer import ABTestAnalyzer

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

def get_active_experiment(experiments: dict):
    for exp_id, exp_data in experiments.items():
        if exp_data.get("status") == "active":
            return exp_id, exp_data
    return None, None


def main():

    logger.info("Starting A/B Testing...")

    # Load Configuration
    ab_config = config["ab_testing"]

    # Checking if A/B testing is enabled
    if not ab_config["enabled"]:
        logger.info("A/B testing is disabled in config.toml")
        logger.info("Set [ab_testing] enabled = true to run experiments")
        return
    
    # Check for active experiment
    experiments = ab_config["experiments"]
    exp_id, exp = get_active_experiment(experiments)
    # print(exp_id, exp)

    if not exp:
        logger.info("No active experiment found")
        logger.info("Configure an experiment with status = 'active' in config.toml")
        return

    logger.info(f"Running experiment: {exp['name']}!")
    logger.info(f"ID: {exp_id}")
    logger.info(f"Variants: {list(exp['variants'].keys())}")

    # Load and prepare data
    train_df = load_data()
    processed_df = create_features(train_df)
    logger.info(f"Total samples: {len(processed_df)}")

    # Split data
    logger.info("Splitting data...")
    splitter = ABTestDataSplitter(ab_config["data_split"])
    train_split, val_split, ab_test_split = splitter.split(processed_df)

    logger.info(f"Training set: {len(train_split)} samples ({len(train_split)/len(processed_df)*100:.1f}%)")
    logger.info(f"Validation set: {len(val_split)} samples ({len(val_split)/len(processed_df)*100:.1f}%)")
    logger.info(f"A/B Test set: {len(ab_test_split)} samples ({len(ab_test_split)/len(processed_df)*100:.1f}%)")

    logger.info("Loading models...")
    models = load_models()

    if len(models) < 2:
        logger.info("Need both models to run A/B test")
        logger.info("Run 'python main_train.py' first to train models")
        return
    
    # Create prediction service
    service = ABTestPredictionService(models=models,feature_columns=config["features"], 
                                      ab_config=ab_config, experiment_id=exp_id)

    # Run experiment
    logger.info(f"Running A/B test on {len(ab_test_split)} samples...")
    results = service.run_experiment(ab_test_data=ab_test_split,
                                     target_column=config["target"])
        
    print(f"Results:")
    print(f"Total predictions: {results['total_predictions']}")
    print(f"Control (RF): {results['control_count']} ({results['control_count']/results['total_predictions']*100:.1f}%)")
    print(f"Treatment (XGB): {results['treatment_count']} ({results['treatment_count']/results['total_predictions']*100:.1f}%)")

    # print("\n📊 Metrics Comparison:")
    metrics = calculate_ab_metrics(exp_id)
    print(metrics)
    comparison = compare_variants(metrics)

    if "error" not in comparison:
        print(f"\n   {'Metric':<15} {'Control (RF)':<15} {'Treatment (XGB)':<15} {'Improvement':<12}")
        print("   " + "-" * 57)

        ctrl = comparison["control"]
        treat = comparison["treatment"]
        comp = comparison["comparison"]

        print(
            f"   {'R² Score':<15} "
            f"{ctrl['r2_score']:<15.4f} "
            f"{treat['r2_score']:<15.4f} "
            f"{comp['r2_improvement_%']:>+.2f}%"
        )

        print(
            f"   {'MAE':<15} "
            f"{ctrl['mae']:<15.4f} "
            f"{treat['mae']:<15.4f} "
            f"{comp['mae_improvement_%']:>+.2f}%"
        )

        print(
            f"   {'RMSE':<15} "
            f"{ctrl['rmse']:<15.4f} "
            f"{treat['rmse']:<15.4f} "
            f"{comp['rmse_improvement_%']:>+.2f}%"
        )

        print(
            f"   {'Sample Size':<15} "
            f"{ctrl['sample_size']:<15} "
            f"{treat['sample_size']:<15}"
        )
    else:
        print(comparison["error"])

    print("\n📉 Statistical Analysis:")
    try:
        analyzer = ABTestAnalyzer()
        sig_result = analyzer.analyze_significance(exp_id)

        print(f"P-value: {sig_result.p_value:.4f}")
        print(f"   Significant: {'Yes ✓' if sig_result.is_significant else 'No'}")
        print(f"   Winner: {sig_result.winner or 'None (inconclusive)'}")
        print(f"\n   💡 {sig_result.recommendation}")

    except Exception as e:
        print(f"Could not perform analysis: {e}")

    print(type(sig_result.is_significant))

    decision = {"experiment_id": exp_id,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "is_significant": sig_result.is_significant,
    "winner": sig_result.winner,
    "selected_model": (sig_result.winner
        if sig_result.is_significant and sig_result.winner
        else "control")}

    os.makedirs("evaluation", exist_ok=True)

    with open("evaluation/model_promotion.json", "w") as f:
        json.dump(decision, f)

    save_metrics(comparison, exp_id)

if __name__ == "__main__":
    main()