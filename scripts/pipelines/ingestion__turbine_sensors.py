# Databricks notebook source
import dlt
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, current_timestamp, to_timestamp

# COMMAND ----------
landing_root_path = spark.conf.get("data.landing_root_path", "")
turbine_sensors_source_files = f"{landing_root_path}/turbine/*.csv"


# COMMAND ----------
@dlt.table(name="bronze.turbine_sensors")
def bronze_turbine() -> DataFrame:
    """Autoloade the turbine sensors data from the landing zone."""
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .load(turbine_sensors_source_files)
        .withColumn("_bronze_laod_timestamp", current_timestamp())
        .withColumn("_file_metadata", col("_metadata"))
    )


# COMMAND ----------
@dlt.table(name="silver.turbine_sensors_cleaned")
@dlt.expect("not_empty_turbine_id", "turbine_id IS NOT NULL")
@dlt.expect("valid_wind_direction", "wind_direction BETWEEN 0 AND 359")
@dlt.expect("valid_wind_speed", "wind_speed >= 0")
@dlt.expect("valid_power_output", "power_output >= 0")
def silver_turbine_sensors_cleaned() -> DataFrame:
    """Clean the turbine sensors data using DLT expectations mechanism."""
    return dlt.readStream("bronze.turbine_sensors").select(
        to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss").alias("timestamp"),
        col("turbine_id"),
        col("wind_speed").cast("double").alias("wind_speed"),
        col("wind_direction").cast("int").alias("wind_direction"),
        col("power_output").cast("double").alias("power_output"),
    )
