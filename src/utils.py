from pyspark.sql import DataFrame
from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
from pyspark.sql import functions as spf


def check_key_cols(
    df: DataFrame | ConnectDataFrame,
    key_cols: list[str],
) -> bool:
    """
    Validate key columns in the DataFrame or ConnectDataFrame.

    :param df: The input DataFrame or ConnectDataFrame.
    :param key_cols: List of key col names.
    :return: True if all key columns are present in the DataFrame or ConnectDataFrame, False otherwise.
    """

    # Check if key columns as composite key represent unique records
    composite_key_check_df = df.groupBy(key_cols).count().filter("count > 1")

    # If there are duplicate records, the count will not be 0
    return composite_key_check_df.count() == 0


def combine_composite_key(
    df: DataFrame | ConnectDataFrame,
    key_cols: list[str],
    composite_key_col: str,
) -> DataFrame | ConnectDataFrame:
    """
    Add a composite key column to the DataFrame.

    Args:
        spark_session (SparkSession): The Spark session.
        df (DataFrame | ConnectDataFrame): The DataFrame to add the composite key to.
        key_cols (list): The list of columns to use for the composite key.
        composite_key_col (str): The name of the composite key column.

    Returns:
        DataFrame | ConnectDataFrame: The DataFrame with the composite key column added.
    """

    return df.withColumn(composite_key_col, spf.concat_ws("_", *key_cols))


def get_unique_col_name(cols: list[str], name_hint: str) -> str:
    """
    Generate a unique column name based on the provided list of columns and a name hint.

    Args:
        cols (list): The list of existing column names.
        name_hint (str): The hint for the new column name.

    Returns:
        str: A unique column name.
    """

    if name_hint not in cols:
        return name_hint

    i = 1
    while f"{name_hint}_{i}" in cols:
        i += 1

    return f"{name_hint}_{i}"
