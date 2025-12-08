from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

def xgb_regressor(df):

    x_df = df[["year",	"day" , "hour", "season", "holiday",	"workingday", "weather", "temp",
               	"humidity", "windspeed", "casual",	"registered"]]
    y_df = df[["count"]]
    
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)
    
    print("Fitting XGBoost Regressor....")
    xgb_regr = XGBRegressor(n_estimators=300, max_depth=5, eta=0.05, random_state=42)
    xgb_regr.fit(x_train, y_train)
    
    count_pred = xgb_regr.predict(x_test)
    
    score = r2_score(y_test, count_pred)
    mae = mean_absolute_error(y_test, count_pred)
    rmse = root_mean_squared_error(y_test, count_pred)
    
    print("Evaluation metric over test dataset....")
    print("R2:", score)
    print("RMSE:", rmse)
    print("MAE:", mae)
    
    count_pred = xgb_regr.predict(x_train)
    
    score = r2_score(y_train, count_pred)
    mae = mean_absolute_error(y_train, count_pred)
    rmse = root_mean_squared_error(y_train, count_pred)
    
    print("Evaluation metric over train dataset....\n")
    print("R2:", score)
    print("RMSE:", rmse)
    print("MAE:", mae)

def rf_regressor(df):
    
    x_df = df[["year",	"day" , "hour", "season", "holiday",	"workingday", "weather", "temp",
               	"humidity", "windspeed", "casual",	"registered"]]
    y_df = df[["count"]]
    
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)
    
    print("Fitting Random Forest Regressor....")
    rf_regr = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)
    rf_regr.fit(x_train, y_train)
    
    count_pred = rf_regr.predict(x_test)
    
    score = r2_score(y_test, count_pred)
    mae = mean_absolute_error(y_test, count_pred)
    rmse = root_mean_squared_error(y_test, count_pred)
    
    print("Evaluation metric over test dataset....")
    print("R2:", score)
    print("RMSE:", rmse)
    print("MAE:", mae)
    
    count_pred = rf_regr.predict(x_train)
    
    score = r2_score(y_train, count_pred)
    mae = mean_absolute_error(y_train, count_pred)
    rmse = root_mean_squared_error(y_train, count_pred)
    
    print("Evaluation metric over train dataset....")
    print("R2:", score)
    print("RMSE:", rmse)
    print("MAE:", mae)