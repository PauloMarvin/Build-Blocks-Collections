import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricUtils:
    @staticmethod
    def compute_accuracy(y_pred: pd.DataFrame, y_true: pd.DataFrame, **kwargs) -> float:
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred, **kwargs)
        return float(accuracy)

    @staticmethod
    def compute_precision(
        y_pred: pd.DataFrame,
        y_true: pd.DataFrame,
        average: str = "binary",
        **kwargs,
    ) -> float:
        precision = precision_score(
            y_true=y_true, y_pred=y_pred, average=average, **kwargs
        )
        return float(precision)

    @staticmethod
    def compute_recall(
        y_pred: pd.DataFrame,
        y_true: pd.DataFrame,
        average: str = "binary",
        **kwargs,
    ) -> float:
        recall = recall_score(y_true=y_true, y_pred=y_pred, average=average, **kwargs)
        return float(recall)

    @staticmethod
    def compute_f1_score(
        y_pred: pd.DataFrame,
        y_true: pd.DataFrame,
        average: str = "binary",
        **kwargs,
    ) -> float:
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average, **kwargs)
        return float(f1)

    @staticmethod
    def compute_confusion_matrix(
        y_pred: pd.DataFrame, y_true: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, **kwargs)
        return pd.DataFrame(
            matrix,
            columns=["Predicted Negative", "Predicted Positive"],
            index=["Actual Negative", "Actual Positive"],
        )

    @staticmethod
    def generate_classification_report(
        y_pred: pd.DataFrame, y_true: pd.DataFrame, **kwargs
    ) -> str:
        report = classification_report(y_true=y_true, y_pred=y_pred, **kwargs)
        return report
