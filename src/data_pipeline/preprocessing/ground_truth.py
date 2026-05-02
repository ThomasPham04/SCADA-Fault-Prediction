"""
GroundTruth — data_pipeline.preprocessing.ground_truth

Creates row-level ground truth labels from event_info metadata.

Label definition (based on event_info.csv, NOT status_type_id):
    0  ->  Normal
    1  ->  Fault   (event_start_id <= id <= event_end_id, anomaly events only)

For normal events, every row receives label=0.

Used for:
  - Evaluating anomaly detection models (Option A: unsupervised autoencoder)
  - Training supervised classifiers    (Option B: binary classifier)
"""

import pandas as pd


class GroundTruth:
    """
    Produces row-level ground truth labels for CARE to Compare event datasets.

    Args:
        event_info: DataFrame from EventLoader.load_event_info().
                    Must contain: event_id, event_label,
                                  event_start_id, event_end_id.
    """

    def __init__(self, event_info: pd.DataFrame) -> None:
        required = {"event_id", "event_label", "event_start_id", "event_end_id"}
        missing = required - set(event_info.columns)
        if missing:
            raise ValueError(f"event_info is missing columns: {missing}")
        self._info = event_info.set_index("event_id")

    # ------------------------------------------------------------------
    # Core label creation
    # ------------------------------------------------------------------

    def make_labels(self, df: pd.DataFrame, event_id: int) -> pd.Series:
        """
        Integer labels (0 = normal, 1 = fault) for a single event dataset.

        For anomaly events: rows where event_start_id <= id <= event_end_id
        receive label=1.  All other rows receive label=0.
        For normal events: every row receives label=0.

        Args:
            df:       Full event DataFrame — must have an 'id' column.
            event_id: Numeric event identifier matching event_info.

        Returns:
            pd.Series[int] with the same index as df, named 'label'.
        """
        event  = self._info.loc[event_id]
        labels = pd.Series(0, index=df.index, dtype=int, name="label")

        if event["event_label"] == "anomaly":
            fault_mask = (
                (df["id"] >= event["event_start_id"]) &
                (df["id"] <= event["event_end_id"])
            )
            labels[fault_mask] = 1

        return labels

    def make_normal_index(self, df: pd.DataFrame, event_id: int) -> pd.Series:
        """
        Boolean normal_index matching the authors' convention.

        True  = normal row  (label == 0)
        False = fault row   (label == 1)

        This mirrors care2compare.py's normal_index used for training
        the autoencoder (train only on normal rows).

        Args:
            df:       Full event DataFrame — must have an 'id' column.
            event_id: Numeric event identifier.

        Returns:
            pd.Series[bool] — True means normal, False means fault.
        """
        return self.make_labels(df, event_id) == 0

    # ------------------------------------------------------------------
    # Convenience: add label column directly to a DataFrame copy
    # ------------------------------------------------------------------

    def add_label_column(self, df: pd.DataFrame, event_id: int) -> pd.DataFrame:
        """
        Return a copy of df with a 'label' column added.

        Args:
            df:       Event DataFrame — must have an 'id' column.
            event_id: Numeric event identifier.

        Returns:
            DataFrame copy with 'label' column appended.
        """
        out = df.copy()
        out["label"] = self.make_labels(df, event_id).values
        return out

    # ------------------------------------------------------------------
    # Convenience: label summary
    # ------------------------------------------------------------------

    def summary(self, df: pd.DataFrame, event_id: int) -> None:
        """Print label distribution for an event dataset."""
        event  = self._info.loc[event_id]
        labels = self.make_labels(df, event_id)
        n_fault  = labels.sum()
        n_normal = len(labels) - n_fault
        print(
            f"Event {event_id} ({event['event_label']:7s}) | "
            f"Normal={n_normal:,}  Fault={n_fault:,}  Total={len(labels):,}"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def make_labels(df: pd.DataFrame, event_id: int, event_info: pd.DataFrame) -> pd.Series:
    """Module-level alias for GroundTruth.make_labels()."""
    return GroundTruth(event_info).make_labels(df, event_id)


def make_normal_index(df: pd.DataFrame, event_id: int, event_info: pd.DataFrame) -> pd.Series:
    """Module-level alias for GroundTruth.make_normal_index()."""
    return GroundTruth(event_info).make_normal_index(df, event_id)


def add_label_column(df: pd.DataFrame, event_id: int, event_info: pd.DataFrame) -> pd.DataFrame:
    """Module-level alias for GroundTruth.add_label_column()."""
    return GroundTruth(event_info).add_label_column(df, event_id)
