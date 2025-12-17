from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from typing import Dict, List
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass

class SklearnClassifier():

    def __init__(self, estimator: BaseEstimator, features: List[str], target: str,):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        print(df_train[self.features].columns)
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame):

        pred= self.predict(df_test)
        true = df_test[self.target].values
        
        score = r2_score(true, pred)
        mae = mean_absolute_error(true, pred)
        rmse = root_mean_squared_error(true, pred)

        metrics = {"R2 score": score,
                   "MAE": mae,
                   "RMSE": rmse}
        
        return metrics
    
    def predict(self, df: pd.DataFrame):
        return self.clf.predict(df[self.features].values)
