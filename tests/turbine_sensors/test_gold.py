from datetime import datetime

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.testing import assertDataFrameEqual

from colibri.turbine_sensors.gold import turbine_sensors_summarize


@pytest.fixture
def spark():
    """Create a SparkSession for testing."""
    return SparkSession.builder.master().getOrCreate()


@pytest.fixture
def sample_data(spark):
    """Create a sample DataFrame for testing."""
    schema = StructType(
        [
            StructField("turbine_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("power_output", DoubleType(), True),
        ]
    )
    data = [
        Row(turbine_id="1", timestamp=datetime(2024, 1, 1, 0, 0, 0), power_output=1.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 1, 1, 0, 0), power_output=4.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 1, 2, 0, 0), power_output=5.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 1, 3, 0, 0), power_output=3.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 1, 4, 0, 0), power_output=2.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 5, 0, 0, 0), power_output=0.5),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 5, 1, 0, 0), power_output=4.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 5, 2, 0, 0), power_output=10.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 5, 3, 0, 0), power_output=3.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 5, 4, 0, 0), power_output=2.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 20, 0, 0, 0), power_output=0.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 20, 1, 0, 0), power_output=4.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 20, 2, 0, 0), power_output=5.0),
        Row(turbine_id="1", timestamp=datetime(2024, 1, 20, 3, 0, 0), power_output=3.0),
        Row(
            turbine_id="1", timestamp=datetime(2024, 1, 20, 4, 0, 0), power_output=15.0
        ),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def expected_schema():
    """Create a sample schema for testing."""
    return StructType(
        [
            StructField("period_start_date", TimestampType(), True),
            StructField("turbine_id", StringType(), True),
            StructField("min_power_output", DoubleType(), True),
            StructField("max_power_output", DoubleType(), True),
            StructField("avg_power_output", DoubleType(), True),
            StructField("period_granularity", StringType(), True),
        ]
    )


def test_turbine_sensors_summarize_daily(spark, sample_data, expected_schema):
    """Test the turbine_sensors_summarize function."""
    result_df = turbine_sensors_summarize(sample_data, "day")

    expected_data = [
        Row(
            period_start_date=datetime(2024, 1, 1, 0, 0, 0),
            turbine_id="1",
            min_power_output=1.0,
            max_power_output=5.0,
            avg_power_output=3.0,
            period_granularity="day",
        ),
        Row(
            period_start_date=datetime(2024, 1, 5, 0, 0, 0),
            turbine_id="1",
            min_power_output=0.5,
            max_power_output=10.0,
            avg_power_output=3.9,
            period_granularity="day",
        ),
        Row(
            period_start_date=datetime(2024, 1, 20, 0, 0, 0),
            turbine_id="1",
            min_power_output=0.0,
            max_power_output=15.0,
            avg_power_output=5.4,
            period_granularity="day",
        ),
    ]

    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assertDataFrameEqual(result_df, expected_df)


def test_turbine_sensors_summarize_weekly(spark, sample_data, expected_schema):
    """Test the turbine_sensors_summarize function."""
    result_df = turbine_sensors_summarize(sample_data, "week")

    expected_data = [
        Row(
            period_start_date=datetime(2024, 1, 1, 0, 0, 0),
            turbine_id="1",
            min_power_output=0.5,
            max_power_output=10.0,
            avg_power_output=3.45,
            period_granularity="week",
        ),
        Row(
            period_start_date=datetime(2024, 1, 15, 0, 0, 0),
            turbine_id="1",
            min_power_output=0.0,
            max_power_output=15.0,
            avg_power_output=5.4,
            period_granularity="week",
        ),
    ]

    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assertDataFrameEqual(result_df, expected_df)


def test_turbine_sensors_summarize_monthly(spark, sample_data, expected_schema):
    """Test the turbine_sensors_summarize function."""
    result_df = turbine_sensors_summarize(sample_data, "month")

    expected_data = [
        Row(
            period_start_date=datetime(2024, 1, 1, 0, 0, 0),
            turbine_id="1",
            min_power_output=0.0,
            max_power_output=15.0,
            avg_power_output=4.1,
            period_granularity="month",
        ),
    ]

    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assertDataFrameEqual(result_df, expected_df)
