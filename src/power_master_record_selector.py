from dataclasses import dataclass, field
from typing import ClassVar
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
from pyspark.sql.connect.session import SparkSession as ConnectSparkSession
from .master_record_selector import MasterRecordSelector


@dataclass
class PowerMasterRecordSelector(MasterRecordSelector):

    DEFAULT_POWER: ClassVar[int] = 1

    powers: dict[str, int] = field(default_factory=dict)

    def select_masters(
        self,
        spark_session: SparkSession | ConnectSparkSession,
        df: DataFrame | ConnectDataFrame,
        cluster_col: str,
        key_cols: list[str],
        master_col_prefix: str = "master_",
    ) -> DataFrame | ConnectDataFrame:
        """
        Abstract method to select master records from a DataFrame or ConnectDataFrame.

        :param df: The input DataFrame or ConnectDataFrame.
        :param cluster_col: The col name representing the cluster.
        :param key_cols: List of key col names.
        :return: A DataFrame or ConnectDataFrame with master records selected.
        """

        default_power = self.powers.get("*", self.DEFAULT_POWER)
        non_key_cols = [col for col in df.columns if col not in key_cols]

        # Sql query to pick a master record
        master_sql = f"""
            with power as (
                select
                    { ", ".join(key_cols) }
                    , {cluster_col}
                    , { " + ".join([
                        f"(case when {col} is not null then {self.powers.get(col, default_power)} else 0 end)"
                        for col in df.columns if col != cluster_col
                    ]) } as power
                from {{cluster}}
            ),
            ranks as (
                select
                    { ", ".join(key_cols) }
                    , {cluster_col}
                    , row_number() over (partition by {cluster_col} order by power desc) as rank
                from power
            )

            select
                { ", ".join([f"{{cluster}}.{col}" for col in key_cols]) }
                , { ", ".join([f"ranks.{col} as {master_col_prefix}{col}" for col in key_cols]) }
                , { ", ".join([f"{{cluster}}.{col}" for col in non_key_cols]) }
            from {{cluster}}
            left join ranks on {{cluster}}.{cluster_col} = ranks.{cluster_col} and ranks.rank = 1
        """

        master_df = spark_session.sql(
            master_sql,
            cluster=df,
        )

        return master_df
