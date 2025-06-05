from typing import Callable
from dataclasses import dataclass
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
from pyspark.sql.connect.session import SparkSession as ConnectSparkSession
from splink import DuckDBAPI, Linker, SettingsCreator
from .master_record_selector import MasterRecordSelector
from . import utils


@dataclass
class Result:
    result: DataFrame | ConnectDataFrame
    get_analysis: Callable[[], DataFrame | ConnectDataFrame]


def generate(
    spark_session: SparkSession | ConnectSparkSession,
    df: DataFrame | ConnectDataFrame,
    key_cols: list[str],
    blocking_rules: list,
    comparisions: list,
    master_record_selector: MasterRecordSelector,
    db_api=DuckDBAPI(),
    threshold_match_probability=0.9,
    master_col_prefix: str = "master_",
) -> Result:
    """
    Generate master records using the PowerMasterRecordSelector.

    :param spark: Spark session.
    :param df: Source DataFrame or ConnectDataFrame.
    :param cluster_col: Column name representing the cluster.
    :param key_cols: List of key col names.
    :param powers: Dictionary of col powers.
    :param master_col_prefix: Prefix for master col names.
    :return: DataFrame or ConnectDataFrame with master records selected.
    """

    assert isinstance(
        spark_session, (SparkSession, ConnectSparkSession)
    ), "spark_session must be a SparkSession or ConnectSparkSession instance"

    assert isinstance(
        df, (DataFrame, ConnectDataFrame)
    ), "df must be a DataFrame or ConnectDataFrame instance"

    assert isinstance(key_cols, list) and all(
        isinstance(col, str) for col in key_cols
    ), "key_cols must be a list of strings"

    assert len(key_cols) > 0, "key_cols must contain at least one column"

    assert isinstance(blocking_rules, list), "blocking_rules must be a list"

    assert isinstance(comparisions, list), "comparisions must be a list"

    assert isinstance(
        master_record_selector, MasterRecordSelector
    ), "master_record_selector must be an instance of MasterRecordSelector"

    assert isinstance(
        threshold_match_probability, (float, int)
    ), "threshold_match_probability must be a float or int"

    assert isinstance(master_col_prefix,
                      str), "master_col_prefix must be a string"

    assert utils.check_key_cols(
        df, key_cols), "key_cols must represent unique records"

    # --------------------------------------------------------------------------
    # Generate a unique id col for splink, splink does not support
    # composite keys
    # --------------------------------------------------------------------------

    if len(key_cols) == 1:
        unique_id_col_name = key_cols[0]
    else:
        unique_id_col_name = utils.get_unique_col_name(
            cols=df.columns,
            name_hint="composite_key",
        )

        df = utils.combine_composite_key(
            df=df,
            key_cols=key_cols,
            composite_key_col=unique_id_col_name,
        )

    # --------------------------------------------------------------------------
    # Prepare inputs for the model
    # --------------------------------------------------------------------------

    source_pandas_df = df.toPandas()

    # Defining the match configuration
    settings = SettingsCreator(
        unique_id_column_name=unique_id_col_name,
        link_type="dedupe_only",
        comparisons=comparisions,
        blocking_rules_to_generate_predictions=blocking_rules,
    )

    if db_api is None:
        db_api = DuckDBAPI()

    # Create the linker to perform training and prediction
    linker = Linker(source_pandas_df, settings, db_api)

    # --------------------------------------------------------------------------
    # Perform matches
    # --------------------------------------------------------------------------

    # Make predictions
    prediction_splink_df = linker.inference.predict()

    # Create clusters based on the predictions
    cluster_splink_df = linker.clustering.cluster_pairwise_predictions_at_threshold(
        prediction_splink_df,
        threshold_match_probability=threshold_match_probability,
    )

    # Saving the result as spark dataframe
    cluster_pandas_df = cluster_splink_df.as_pandas_dataframe()
    cluster_df = spark_session.createDataFrame(cluster_pandas_df)

    master_df = cluster_df

    # We no longer need the composite key, so drop it if created before
    # for splink
    if len(key_cols) > 1:
        cluster_df = cluster_df.drop(unique_id_col_name)

    # --------------------------------------------------------------------------
    # Select master records
    # --------------------------------------------------------------------------

    # Select the master records
    master_df = master_record_selector.select_masters(
        spark_session=spark_session,
        df=cluster_df,
        key_cols=key_cols,
        cluster_col="cluster_id",
        master_col_prefix=master_col_prefix,
    )

    # We no longer need the cluster_id col, as the master records have
    # been selected.
    master_df = master_df.drop("cluster_id")

    # --------------------------------------------------------------------------
    # Get analysis function
    # --------------------------------------------------------------------------

    def get_anaylsis():

        master_cols = [f"{master_col_prefix}{col}" for col in key_cols]
        other_cols = [
            col for col in master_df.columns if col not in (key_cols + master_cols)
        ]

        analysis_sql = f"""
            with master as (
                select *
                from {{master}}
            )

            , counts as (
                select
                    {", ".join(master_cols)}
                    , count(*) as count
                from master
                group by all
            )

            select
                counts.count as cluster_count
                , {", ".join([f"master.{col}" for col in key_cols])}
                , {", ".join([f"master.{col}" for col in master_cols])}
                , {", ".join([f"master.{col}" for col in other_cols])}
            from master
            left join counts
                on {" and ".join([f"master.{col} = counts.{col}" for col in master_cols])}
            order by
                counts.count desc
                , {", ".join([f"master.{col}" for col in master_cols])}
        """

        analysis_df = spark_session.sql(
            analysis_sql,
            master=master_df,
        )

        return analysis_df

    return Result(
        result=master_df,
        get_analysis=get_anaylsis,
    )
