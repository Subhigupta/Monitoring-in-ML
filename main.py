from src.data.make_dataset import load_data
from src.features.build_features import create_features
from src.models.train_model import rf_clf, xgb_clf

# Load training data
train_df = load_data()
print(train_df.shape)

# Conduct feature engineering
processed_train_df = create_features(train_df)
print(processed_train_df.shape, processed_train_df.columns)

# Train Random Forest and Xgboost classifiers
rf_clf(processed_train_df)
xgb_clf(processed_train_df)