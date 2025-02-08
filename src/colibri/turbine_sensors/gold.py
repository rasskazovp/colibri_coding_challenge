from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, col, date_trunc, lit, max, min, to_date


def turbine_sensors_summarize(df: DataFrame, granularity: str) -> DataFrame:
    """Summarize the turbine sensors data by `granularity`.

    :param df: Input turbine sensors DataFrame
    :param granularity: Granularity to summarize the data. Must be 'day', 'week', or 'month'
    :return: DataFrame with summarized data
    """
    if granularity not in ["day", "week", "month"]:
        raise ValueError("Granularity must be 'day', 'week', or 'month'")

    turbine_sensor_summary_df = (
        df.withColumn("date", to_date(col("timestamp")))
        .withColumn("period_start_date", date_trunc(granularity, col("date")))
        .groupBy("period_start_date", "turbine_id")
        .agg(
            min("power_output").alias("min_power_output"),
            max("power_output").alias("max_power_output"),
            avg("power_output").alias("avg_power_output"),
        )
        .withColumn("period_granularity", lit(granularity))
    )
    return turbine_sensor_summary_df
