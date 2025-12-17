from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.store import AssignmentStore

config = load_config()
store = AssignmentStore()

def rf_clf(processed_train_df):

    df_train, df_test = train_test_split(processed_train_df, test_size=config["test_size"])

    rf_estimator = RandomForestClassifier(**config["random_forest"])
    model = SklearnClassifier(rf_estimator, config["features"], config["target"])
    model.train(df_train)

    metrics = model.evaluate(df_test)

    store.put_rf_model("rf_model.onnx", model.clf)
    # store.put_metrics("metrics.json", metrics)

# def xgb_clf(best_params):

#     config = load_config()

#     df = 
#     df_train, df_test = train_test_split(df, test_size=config["test_size"], random_state=42)

#     xgb_estimator = XGBClassifier(**best_params)
#     model = SklearnClassifier(xgb_estimator, config["features"], config["target"])
#     model.train(df_train)

#     metrics = model.evaluate(df_test)

#     store.put_model("saved_model.pkl", model)
#     store.put_metrics("metrics.json", metrics)

# if __name__ == "__main__":
#     #print("Random Forest: Baseline model being trained!")
#     #main()

#     print("Xgboost: Finding the best hyperparameters!")
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=20)

#     best_params = study.best_params
#     # Add fixed params
#     best_params.update({"random_state": 42,
#         "eval_metric": "logloss"})
    
#     print("Best parameters:", best_params)
    
#     xgb_clf(best_params)

