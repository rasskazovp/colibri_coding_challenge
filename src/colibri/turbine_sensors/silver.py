from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    array,
    col,
    explode,
    expr,
    lag,
    lead,
    mean,
    stddev,
    to_date,
    when,
)
from pyspark.sql.window import Window


def turbine_sensors_anomalies_detection(df: DataFrame) -> DataFrame:
    """Detect anomalies in `power_output` of the turbine sensors data.

    :param df: Input turbine sensors DataFrame
    :return: DataFrame with anomaly detection columns
    """
    window_spec = (
        Window.partitionBy("turbine_id")
        .orderBy("timestamp_in_sec")
        .rangeBetween(-86400, 86400)  # -24 hours to +24 hours
    )

    return (
        df.withColumn("timestamp_in_sec", col("timestamp").cast("long"))
        .withColumn("power_output_mean", mean(col("power_output")).over(window_spec))
        .withColumn(
            "power_output_stddev", stddev(col("power_output")).over(window_spec)
        )
        .withColumn(
            "power_output_anomaly",
            (
                col("power_output")
                > col("power_output_mean") + 2 * col("power_output_stddev")
            )
            | (
                col("power_output")
                < col("power_output_mean") - 2 * col("power_output_stddev")
            ),
        )
    )


def turbine_sensors_data_imputation(df: DataFrame) -> DataFrame:
    """Impute missing or invalid values in the turbine sensors data.

    :param df: Input turbine sensors DataFrame
    :return: DataFrame with imputed values
    """
    turbine_sensors_all_rec_df = _turbine_sensors_fill_missing_records(df)

    window_spec = Window.partitionBy("turbine_id").orderBy("timestamp")

    return (
        turbine_sensors_all_rec_df.withColumn(
            "prev_power_output", lag("power_output").over(window_spec)
        )
        .withColumn("next_power_output", lead("power_output").over(window_spec))
        .withColumn(
            "imputed_power_output",
            when(col("next_power_output").isNull(), col("prev_power_output"))
            .when(col("prev_power_output").isNull(), col("next_power_output"))
            .otherwise((col("prev_power_output") + col("next_power_output")) / 2),
        )
        .withColumn("power_output_source_value", col("power_output"))
        .withColumn(
            "power_output",
            when(
                col("power_output").isNull() | col("power_output_anomaly"),
                col("imputed_power_output"),
            ).otherwise(col("power_output")),
        )
        .drop("prev_power_output", "next_power_output", "imputed_power_output")
    )


def _turbine_sensors_fill_missing_records(df: DataFrame) -> DataFrame:
    """Fill missing records in the turbine sensors data.

    :param df: Input turbine sensors DataFrame
    :return: DataFrame with filled missing records
    """
    unique_dates_df = df.select(to_date("timestamp").alias("date")).distinct()
    hourly_timestamps_df = unique_dates_df.withColumn(
        "hour_interval", explode(array([expr(f"INTERVAL {h} HOUR") for h in range(24)]))
    ).select(expr("date + hour_interval").alias("timestamp"))

    return hourly_timestamps_df.join(df, on="timestamp", how="left")
