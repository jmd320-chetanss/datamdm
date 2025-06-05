from abc import ABC, abstractmethod
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
from pyspark.sql.connect.session import SparkSession as ConnectSparkSession


class MasterRecordSelector(ABC):

    @abstractmethod
    def select_masters(
        self,
        spark_session: SparkSession | ConnectSparkSession,
        df: DataFrame | ConnectDataFrame,
        cluster_col: str,
        key_cols: list[str],
        master_col_prefix: str = "master_",
    ) -> DataFrame | ConnectDataFrame:
        pass
