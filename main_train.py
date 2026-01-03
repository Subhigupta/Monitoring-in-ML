from src.data.make_dataset import load_data
from src.features.build_features import create_features
from src.models.train_model import rf_clf, xgb_clf
from src.ab_testing.data_splitter import ABTestDataSplitter
from src.utils.logger import get_logger
from src.utils.config import load_config

config = load_config()

logger = get_logger(__name__)

# Load training data
logger.info("Loading Training data...")
train_df = load_data()
print(train_df.shape)

# Conduct feature engineering
logger.info("Conducting Feature Engineering...")
processed_train_df = create_features(train_df)

# Split the data
splitter = ABTestDataSplitter(config["ab_testing"]["data_split"])
train_df, val_df, ab_test_df = splitter.split(processed_train_df)

# Train Random Forest and Xgboost classifiers
rf_clf(train_df, val_df)
xgb_clf(train_df, val_df)

