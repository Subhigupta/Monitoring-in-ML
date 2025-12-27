"""Data splitter for A/B testing holdout set."""

import os
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


class ABTestDataSplitter:
    """Splits train.csv into training, validation, and A/B test sets."""
    
    def __init__(self, config: dict):
        """Initialize with split configuration."""
        self.config = config
        self.train_ratio = config["train_ratio"]
        self.validation_ratio = config["validation_ratio"]
        self.ab_test_ratio = config["ab_test_ratio"]
        self._validate_ratios()
    
    def _validate_ratios(self) -> None:
        """Ensure ratios sum to 1.0."""
        total = self.train_ratio + self.validation_ratio + self.ab_test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train, validation, and A/B test sets.
        
        Args:
            df: Full training dataframe with ground truth labels
            
        Returns:
            Tuple of (train_df, validation_df, ab_test_df)
        """
        
        # First split: separate ab_test set
        remaining_ratio = self.train_ratio + self.validation_ratio
        train_val_df, ab_test_df = train_test_split(
            df,
            test_size=self.ab_test_ratio,
            random_state=self.config["random_state"]
        )
        
        # Second split: separate train and validation
        val_ratio_adjusted = self.validation_ratio / remaining_ratio
        train_df, validation_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=self.config["random_state"]
        )
        
        return train_df, validation_df, ab_test_df
    
    def get_ab_test_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get only the A/B test holdout set."""
        _, _, ab_test_df = self.split(df)
        return ab_test_df
