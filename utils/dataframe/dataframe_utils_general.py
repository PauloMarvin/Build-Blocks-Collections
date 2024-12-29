import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek
from sklearn.utils import resample


class DataframeUtilsGeneral:
    @staticmethod
    def __apply_sampler(
            df: pd.DataFrame, target_col: str, sampler, **sampler_kwargs
    ) -> pd.DataFrame:
        """Helper function to apply given sampler and merge results back."""
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        sampler_instance = sampler(random_state=42, **sampler_kwargs)
        resampled_features, resampled_target = sampler_instance.fit_resample(X, y)
        return pd.concat(
            [pd.DataFrame(resampled_features), pd.DataFrame(resampled_target, columns=[target_col])], axis=1
        )

    @staticmethod
    def random_under_sampling(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Applies Random Under-Sampling."""
        return DataframeUtilsGeneral.__apply_sampler(df, target_col, RandomUnderSampler)

    @staticmethod
    def random_over_sampling(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Applies Random Over-Sampling."""
        return DataframeUtilsGeneral.__apply_sampler(df, target_col, RandomOverSampler)

    @staticmethod
    def smote(df: pd.DataFrame, target_col: str, k_neighbors: int = 5) -> pd.DataFrame:
        """Applies SMOTE."""
        return DataframeUtilsGeneral.__apply_sampler(df, target_col, SMOTE, k_neighbors=k_neighbors)

    @staticmethod
    def borderline_smote(df: pd.DataFrame, target_col: str, k_neighbors: int = 5) -> pd.DataFrame:
        """Applies BorderlineSMOTE."""
        return DataframeUtilsGeneral.__apply_sampler(df, target_col, BorderlineSMOTE, k_neighbors=k_neighbors)

    @staticmethod
    def svm_smote(df: pd.DataFrame, target_col: str, k_neighbors: int = 5) -> pd.DataFrame:
        """Applies SVM-SMOTE."""
        return DataframeUtilsGeneral.__apply_sampler(df, target_col, SVMSMOTE, k_neighbors=k_neighbors)

    @staticmethod
    def adasyn(df: pd.DataFrame, target_col: str, n_neighbors: int = 5) -> pd.DataFrame:
        """Applies ADASYN."""
        return DataframeUtilsGeneral.__apply_sampler(df, target_col, ADASYN, n_neighbors=n_neighbors)

    @staticmethod
    def smote_tomek(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Applies SMOTETomek."""
        return DataframeUtilsGeneral.__apply_sampler(df, target_col, SMOTETomek)

    @staticmethod
    def bootstrap_resampling(df: pd.DataFrame, target_col: str, n_samples: int = None) -> pd.DataFrame:
        """Performs Bootstrap Resampling."""
        if not n_samples:
            n_samples = len(df)
        return resample(df, replace=True, n_samples=n_samples, random_state=42)

    @staticmethod
    def check_class_distribution(df: pd.DataFrame, target_col: str) -> None:
        """Displays the class distribution."""
        print(df[target_col].value_counts(normalize=True) * 100)
