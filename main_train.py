from src.data.make_dataset import load_data
from src.features.build_features import create_features
from src.models.train_model import rf_clf, xgb_clf
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Load training data
logger.info("Loading Training data...")
train_df = load_data()
print(train_df.shape)

# Conduct feature engineering
logger.info("Conducting Feature Engineering...")
processed_train_df = create_features(train_df)

# Train Random Forest and Xgboost classifiers
rf_clf(processed_train_df)
xgb_clf(processed_train_df)