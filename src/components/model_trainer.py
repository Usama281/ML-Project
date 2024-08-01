import sys
import os
from src.utils import save_object, evaluate_models
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                'Linear Regression': LinearRegression(),
                'KNeighbors Regression': KNeighborsRegressor(),
                'Decision Tree Regression': DecisionTreeRegressor(),
                'SVR': SVR(),
                'Random Forest Regression': RandomForestRegressor(),
                'AdaBoost Regression': AdaBoostRegressor(),
                'GradientBoosting Regression': GradientBoostingRegressor(),
                'XGBoost Regression': XGBRegressor(),
                'CatBoost Regression': CatBoostRegressor()
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            best_model_score = max(sorted(model_report.values()))
            if best_model_score<0.6:
                raise CustomException('No best model found')
            
            best_model_name = list(model_report.keys()) [
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info('Found best model both on train and test data')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2


        except Exception as e:
            raise CustomException(e, sys)
        