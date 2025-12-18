from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.store import AssignmentStore

config = load_config()
store = AssignmentStore()

def rf_clf(processed_train_df):

    df_train, df_test = train_test_split(processed_train_df, test_size=config["test_size"], 
                                         random_state=config["random_state"])

    rf_estimator = RandomForestClassifier(**config["random_forest"])
    model = SklearnClassifier(rf_estimator, config["features"], config["target"])
    model.train(df_train)

    metrics = model.evaluate(df_test)

    store.put_rf_model("rf_model.onnx", model.clf)
    store.put_metrics("rf_metrics.json", metrics)

def xgb_clf(processed_train_df):

    df_train, df_test = train_test_split(processed_train_df, test_size=config["test_size"], 
                                         random_state=config["random_state"])

    xgb_estimator = XGBClassifier(**config["xgboost"])
    model = SklearnClassifier(xgb_estimator, config["features"], config["target"])
    model.train(df_train)

    metrics = model.evaluate(df_test)

    store.put_xgb_model("xgb_model.pkl", model.clf)
    store.put_metrics("xgb_metrics.json", metrics)

