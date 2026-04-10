"""
FeatureEngineer — data_pipeline.preprocessing.feature_engineering
Transforms raw SCADA DataFrame columns: angle encoding
feature-column selection, and raw value extraction.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import (
    ANGLE_FEATURES,
    COUNTER_FEATURES,
    FEATURE_COLUMNS,
    EXCLUDE_COLUMNS,
)


class FeatureEngineer:
    """
    Applies feature engineering transformations to raw SCADA DataFrames.

    All methods operate on copies of the input; the original DataFrame is
    never modified in-place.
    """
    @staticmethod
    def wrap_angle_deg(series: pd.Series) -> pd.Series:
        """Wrap raw angles into [-180, 180)."""
        values = pd.to_numeric(series, errors="coerce")
        return ((values + 180.0) % 360.0) - 180.0

    def engineer_angle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert angle features to sin/cos components for angular continuity.
        Drops the original degree columns after conversion.

        Args:
            df: DataFrame with raw angle columns.

        Returns:
            DataFrame with sin/cos columns replacing angle columns.
        """
        df = df.copy()

        for col in ANGLE_FEATURES:
            if col not in df.columns:
                continue

            wrapped = self.wrap_angle_deg(df[col])
            radians = np.radians(wrapped)
            df[f"{col}_sin"] = np.sin(radians)
            df[f"{col}_cos"] = np.cos(radians)

        if "sensor_2_avg" in df.columns:
            relative_direction = self.wrap_angle_deg(df["sensor_2_avg"])
            df["yaw_misalignment_abs"] = relative_direction.abs()

        raw_angle_columns = [col for col in ANGLE_FEATURES if col in df.columns]
        if raw_angle_columns:
            df.drop(columns=raw_angle_columns, inplace=True)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Determine the final feature column list after angle engineering.
        Includes base sensor features plus the sin/cos engineered columns.

        Args:
            df: DataFrame after engineer_angle_features and drop_counter_features.

        Returns:
            Ordered list of feature column names to use for modelling.
        """
        feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]

        for angle_col in ANGLE_FEATURES:
            sin_col = f"{angle_col}_sin"
            cos_col = f"{angle_col}_cos"
            if sin_col in df.columns:
                feature_cols.append(sin_col)
            if cos_col in df.columns:
                feature_cols.append(cos_col)

        exclude_all = (
            EXCLUDE_COLUMNS + ["status_type_id"] + ANGLE_FEATURES 
        )
        feature_cols = [col for col in feature_cols if col not in exclude_all]
        return feature_cols

    def preprocess_features(
        self,
        df: pd.DataFrame,
        feature_cols: list,
    ) -> np.ndarray:
        """
        Extract and clean feature values from a DataFrame.
        Applies forward-fill, back-fill, then zero-fill for remaining NaNs.

        Args:
            df: Source DataFrame.
            feature_cols: Columns to extract.

        Returns:
            2D float array of shape (n_timesteps, n_features).
        """
        features = df[feature_cols].copy()
        features = features.ffill().bfill()
        features = features.fillna(0)
        return features.values
