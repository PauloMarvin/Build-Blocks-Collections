import pandas as pd
from sklearn.utils import resample


class DataframeUtilsGeneral:

    @staticmethod
    def undersampling_df(unbalanced_dataframe: pd.DataFrame, y_target: str) -> pd.DataFrame:
        groups = unbalanced_dataframe.groupby(y_target)
        minority_group = groups.size().min()

        balanced_df = pd.DataFrame()
        for group_name, group in groups:
            balanced_group = resample(group, replace=False, n_samples=minority_group, random_state=42)
            balanced_df = pd.concat([balanced_df, balanced_group])

        return balanced_df.reset_index(drop=True)

