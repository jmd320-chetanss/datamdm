from dataclasses import dataclass, field
from typing import ClassVar
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
from pyspark.sql.connect.session import SparkSession as ConnectSparkSession
from .master_record_selector import MasterRecordSelector


@dataclass
class PowerMasterRecordSelector(MasterRecordSelector):

    @dataclass
    class Range:
        min: int
        max: int
        num_converter: callable

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
        cols = df.columns
        non_key_cols = [col for col in df.columns if col not in key_cols]

        power_cols = [col for col in cols if col not in (key_cols + [cluster_col])]
        # range_power_cols = [
        #     col for col in power_cols
        #     if isinstance(self.powers.get(col, None), PowerMasterRecordSelector.Range)
        # ]
        range_power_cols = ["start_date"]
        static_power_cols = [col for col in power_cols if col not in range_power_cols]

        print(f"{power_cols=}")
        print(f"{static_power_cols=}")
        print(f"{range_power_cols=}")

        # Sql query to pick a master record
        master_sql = f"""

            with clusters as (
                select *
                from {{cluster}}
            )

            -- This cte converts the range columns to measurable values (numbers)
            , measures as (
                select
                    { ", ".join(col for col in cols if col not in range_power_cols) }
                    { ", " if len(range_power_cols) > 0 else "" }
                    { ", ".join(f"{self.powers[col].num_converter(col)} as {col}" for col in range_power_cols) }

                from clusters
            )

            -- This cte adds min max ranges of range columns
            , ranges as (
                select
                    {cluster_col}
                    { ", " if len(range_power_cols) > 0 else "" }
                    { ", ".join(f"min({col}) as min_{col}, max({col}) as max_{col}, max({col}) - min({col}) as range_{col}" for col in range_power_cols) }

                from measures

                group by {cluster_col}
            )

            -- This cte calculates static powers for static power columns
            , static_powers as (
                select
                    { ", ".join(key_cols) }
                    , {cluster_col}
                    { ", " if len(static_power_cols) > 0 else "" }
                    { " , ".join([
                        f"case when {col} is not null then { self.powers.get(col, default_power) } else 0 end as {col}"
                        for col in static_power_cols
                    ]) }
                    { ", " if len(range_power_cols) > 0 else "" }
                    { ", ".join(range_power_cols) }

                from measures
            )

            , range_powers as (
                select
                    { ", ".join(key_cols) }
                    , static_powers.{cluster_col}

                    { ", " if len(static_power_cols) > 0 else "" }
                    { ", ".join(static_power_cols) }

                    { ", " if len(range_power_cols) > 0 else "" }
                    { ", ".join([
                        f"({self.powers[col].min} + (({col} - ranges.min_{col}) / ranges.range_{col}) * {self.powers[col].max - self.powers[col].min}) as {col}"
                        for col in range_power_cols
                    ]) }

                from static_powers

                left join ranges
                    on ranges.{cluster_col} = static_powers.{cluster_col}
            )

            , total_powers as (
                select
                    { ", ".join(key_cols) }
                    , {cluster_col}
                    , { " + ".join(static_power_cols + range_power_cols) } as power

                from range_powers
            )

            , ranks as (
                select
                    { ", ".join(key_cols) }
                    , {cluster_col}
                    , row_number() over (
                        partition by {cluster_col} order by power desc
                    ) as rank

                from total_powers
            )

            , result (
                select
                    { ", ".join([f"clusters.{col}" for col in key_cols]) }
                    , { ", ".join([f"ranks.{col} as {master_col_prefix}{col}" for col in key_cols]) }
                    , { ", ".join([f"clusters.{col}" for col in non_key_cols]) }

                from clusters

                left join ranks
                    on clusters.{cluster_col} = ranks.{cluster_col}
                    and ranks.rank = 1
            )

            select *
            from result
        """

        print(master_sql)

        master_df = spark_session.sql(
            master_sql,
            cluster=df,
        )

        return master_df
