from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.store import AssignmentStore
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()
store = AssignmentStore()

def rf_clf(train_df, val_df):

    df_train, df_test = train_df, val_df

    rf_estimator = RandomForestRegressor(**config["random_forest"])
    model = SklearnClassifier(rf_estimator, config["features"], config["target"])
    logger.info("Fitting Random Forest Regressor...")
    model.train(df_train)

    logger.info("Evaluating Random Forest Regressor...")
    metrics = model.evaluate(df_test)

    logger.info("Saving Random Forest Regressor...")
    store.put_rf_model("rf_model.onnx", model.clf)

    logger.info("Saving Random Forest Regressor Evaluation metrics...")
    store.put_metrics("rf_metrics.json", metrics)

def xgb_clf(train_df, val_df,):

    df_train, df_test = train_df, val_df
    xgb_estimator = XGBRegressor(**config["xgboost"])
    model = SklearnClassifier(xgb_estimator, config["features"], config["target"])
    logger.info("Fitting Xgboost Regressor...")
    model.train(df_train)

    logger.info("Evaluating Xgboost Regressor...")
    metrics = model.evaluate(df_test)

    logger.info("Saving Xgboost Regressor...")
    store.put_xgb_model("xgb_model.onnx", model.clf)

    logger.info("Saving Xgboost Regressor Evaluation metrics...")
    store.put_metrics("xgb_metrics.json", metrics)

