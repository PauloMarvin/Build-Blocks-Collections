import pandas as pd


class DataframeUtilsNLP:
    @staticmethod
    def remove_line_by_text_len(
        dataframe: pd.DataFrame, text_column: str, min_len: int = 5
    ) -> pd.DataFrame:
        dataframe = dataframe.assign(
            number_words=dataframe[text_column].apply(lambda x: len(x.split(" ")))
        )
        dataframe = dataframe.drop(dataframe[dataframe.number_words < min_len].index)

        return dataframe.reset_index(drop=True)
