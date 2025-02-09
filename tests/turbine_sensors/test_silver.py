from datetime import datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.testing import assertDataFrameEqual

from colibri.turbine_sensors.silver import (
    _turbine_sensors_fill_missing_records,
    turbine_sensors_anomalies_detection,
    turbine_sensors_data_imputation,
)


@pytest.fixture
def spark():
    """Create a SparkSession for testing."""
    return SparkSession.builder.getOrCreate()


@pytest.fixture
def anomaly_detection_schema():
    """Create a sample anomaly detection schema for testing."""
    return StructType(
        [
            StructField("turbine_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("power_output", DoubleType(), True),
            StructField("timestamp_in_sec", LongType(), True),
            StructField("power_output_mean", DoubleType(), True),
            StructField("power_output_stddev", DoubleType(), True),
            StructField("power_output_anomaly", BooleanType(), True),
        ]
    )


@pytest.fixture
def sample_data_with_issues(spark):
    """Create a sample DataFrame with missing records and `power_output` outliers."""
    schema = StructType(
        [
            StructField("turbine_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("power_output", DoubleType(), True),
        ]
    )
    data = [
        ["1", datetime(2024, 1, 1, 0, 0, 0), 2.7],
        ["1", datetime(2024, 1, 1, 1, 0, 0), 4.4],
        ["1", datetime(2024, 1, 1, 2, 0, 0), 2.9],
        ["1", datetime(2024, 1, 1, 3, 0, 0), 1.8],
        ["1", datetime(2024, 1, 1, 4, 0, 0), 2.3],
        ["1", datetime(2024, 1, 1, 5, 0, 0), 2.2],
        ["1", datetime(2024, 1, 1, 6, 0, 0), 4.2],
        ["1", datetime(2024, 1, 1, 8, 0, 0), 1.6],
        ["1", datetime(2024, 1, 1, 9, 0, 0), 2.7],
        ["1", datetime(2024, 1, 1, 10, 0, 0), 2.8],
        ["1", datetime(2024, 1, 1, 11, 0, 0), 4.0],
        ["1", datetime(2024, 1, 1, 12, 0, 0), 2.0],
        ["1", datetime(2024, 1, 1, 13, 0, 0), 53.9],
        ["1", datetime(2024, 1, 1, 14, 0, 0), 3.9],
        ["1", datetime(2024, 1, 1, 15, 0, 0), 2.9],
        ["1", datetime(2024, 1, 1, 16, 0, 0), 2.3],
        ["1", datetime(2024, 1, 1, 17, 0, 0), 4.3],
        ["1", datetime(2024, 1, 1, 18, 0, 0), 1.8],
        ["1", datetime(2024, 1, 1, 19, 0, 0), 2.5],
        ["1", datetime(2024, 1, 1, 20, 0, 0), 3.1],
        ["1", datetime(2024, 1, 1, 22, 0, 0), 2.2],
        ["1", datetime(2024, 1, 1, 23, 0, 0), 4.4],
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_data_clean(spark):
    """Create a sample DataFrame for testing with clean data."""
    schema = StructType(
        [
            StructField("turbine_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("power_output", DoubleType(), True),
        ]
    )
    data = [
        ["1", datetime(2024, 1, 1, 0, 0, 0), 2.7],
        ["1", datetime(2024, 1, 1, 1, 0, 0), 4.4],
        ["1", datetime(2024, 1, 1, 2, 0, 0), 2.9],
        ["1", datetime(2024, 1, 1, 3, 0, 0), 1.8],
        ["1", datetime(2024, 1, 1, 4, 0, 0), 2.3],
        ["1", datetime(2024, 1, 1, 5, 0, 0), 2.2],
        ["1", datetime(2024, 1, 1, 6, 0, 0), 4.2],
        ["1", datetime(2024, 1, 1, 7, 0, 0), 4.0],
        ["1", datetime(2024, 1, 1, 8, 0, 0), 1.6],
        ["1", datetime(2024, 1, 1, 9, 0, 0), 2.7],
        ["1", datetime(2024, 1, 1, 10, 0, 0), 2.8],
        ["1", datetime(2024, 1, 1, 11, 0, 0), 4.0],
        ["1", datetime(2024, 1, 1, 12, 0, 0), 2.0],
        ["1", datetime(2024, 1, 1, 13, 0, 0), 3.9],
        ["1", datetime(2024, 1, 1, 14, 0, 0), 3.9],
        ["1", datetime(2024, 1, 1, 15, 0, 0), 2.9],
        ["1", datetime(2024, 1, 1, 16, 0, 0), 2.3],
        ["1", datetime(2024, 1, 1, 17, 0, 0), 4.3],
        ["1", datetime(2024, 1, 1, 18, 0, 0), 1.8],
        ["1", datetime(2024, 1, 1, 19, 0, 0), 2.5],
        ["1", datetime(2024, 1, 1, 20, 0, 0), 3.1],
        ["1", datetime(2024, 1, 1, 21, 0, 0), 2.5],
        ["1", datetime(2024, 1, 1, 22, 0, 0), 2.2],
        ["1", datetime(2024, 1, 1, 23, 0, 0), 4.4],
    ]
    return spark.createDataFrame(data, schema)


def test_turbine_sensors_no_anomalies_detection(
    spark, sample_data_clean, anomaly_detection_schema
):
    """Test the turbine_sensors_anomalies_detection function (no anomalies)."""
    result_df = turbine_sensors_anomalies_detection(sample_data_clean)

    expected_data = [
        [
            "1",
            datetime(2024, 1, 1, 0, 0, 0),
            2.7,
            1704067200,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 1, 0, 0),
            4.4,
            1704070800,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 2, 0, 0),
            2.9,
            1704074400,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 3, 0, 0),
            1.8,
            1704078000,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 4, 0, 0),
            2.3,
            1704081600,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 5, 0, 0),
            2.2,
            1704085200,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 6, 0, 0),
            4.2,
            1704088800,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 7, 0, 0),
            4.0,
            1704092400,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 8, 0, 0),
            1.6,
            1704096000,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 9, 0, 0),
            2.7,
            1704099600,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 10, 0, 0),
            2.8,
            1704103200,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 11, 0, 0),
            4.0,
            1704106800,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 12, 0, 0),
            2.0,
            1704110400,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 13, 0, 0),
            3.9,
            1704114000,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 14, 0, 0),
            3.9,
            1704117600,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 15, 0, 0),
            2.9,
            1704121200,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 16, 0, 0),
            2.3,
            1704124800,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 17, 0, 0),
            4.3,
            1704128400,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 18, 0, 0),
            1.8,
            1704132000,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 19, 0, 0),
            2.5,
            1704135600,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 20, 0, 0),
            3.1,
            1704139200,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 21, 0, 0),
            2.5,
            1704142800,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 22, 0, 0),
            2.2,
            1704146400,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 23, 0, 0),
            4.4,
            1704150000,
            2.9749999999999996,
            0.9208927615422274,
            False,
        ],
    ]

    expected_df = spark.createDataFrame(expected_data, anomaly_detection_schema)

    assertDataFrameEqual(result_df, expected_df)


def test_turbine_sensors_anomalies_detection(
    spark, sample_data_with_issues, anomaly_detection_schema
):
    """Test the turbine_sensors_anomalies_detection function."""
    result_df = turbine_sensors_anomalies_detection(sample_data_with_issues)

    expected_data = [
        [
            "1",
            datetime(2024, 1, 1, 0, 0, 0),
            2.7,
            1704067200,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 1, 0, 0),
            4.4,
            1704070800,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 2, 0, 0),
            2.9,
            1704074400,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 3, 0, 0),
            1.8,
            1704078000,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 4, 0, 0),
            2.3,
            1704081600,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 5, 0, 0),
            2.2,
            1704085200,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 6, 0, 0),
            4.2,
            1704088800,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 8, 0, 0),
            1.6,
            1704096000,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 9, 0, 0),
            2.7,
            1704099600,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 10, 0, 0),
            2.8,
            1704103200,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 11, 0, 0),
            4.0,
            1704106800,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 12, 0, 0),
            2.0,
            1704110400,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 13, 0, 0),
            53.9,
            1704114000,
            5.222727272727273,
            10.909974982214859,
            True,
        ],
        [
            "1",
            datetime(2024, 1, 1, 14, 0, 0),
            3.9,
            1704117600,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 15, 0, 0),
            2.9,
            1704121200,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 16, 0, 0),
            2.3,
            1704124800,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 17, 0, 0),
            4.3,
            1704128400,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 18, 0, 0),
            1.8,
            1704132000,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 19, 0, 0),
            2.5,
            1704135600,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 20, 0, 0),
            3.1,
            1704139200,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 22, 0, 0),
            2.2,
            1704146400,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
        [
            "1",
            datetime(2024, 1, 1, 23, 0, 0),
            4.4,
            1704150000,
            5.222727272727273,
            10.909974982214859,
            False,
        ],
    ]

    expected_df = spark.createDataFrame(expected_data, anomaly_detection_schema)

    assertDataFrameEqual(result_df, expected_df)


def test_turbine_sensors_fill_missing_records(spark, sample_data_with_issues):
    """Test the _turbine_sensors_fill_missing_records function."""
    result_df = _turbine_sensors_fill_missing_records(sample_data_with_issues)

    expected_schema = StructType(
        [
            StructField("turbine_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("power_output", DoubleType(), True),
        ]
    )

    expected_data = [
        ["1", datetime(2024, 1, 1, 0, 0, 0), 2.7],
        ["1", datetime(2024, 1, 1, 1, 0, 0), 4.4],
        ["1", datetime(2024, 1, 1, 2, 0, 0), 2.9],
        ["1", datetime(2024, 1, 1, 3, 0, 0), 1.8],
        ["1", datetime(2024, 1, 1, 4, 0, 0), 2.3],
        ["1", datetime(2024, 1, 1, 5, 0, 0), 2.2],
        ["1", datetime(2024, 1, 1, 6, 0, 0), 4.2],
        ["1", datetime(2024, 1, 1, 7, 0, 0), None],
        ["1", datetime(2024, 1, 1, 8, 0, 0), 1.6],
        ["1", datetime(2024, 1, 1, 9, 0, 0), 2.7],
        ["1", datetime(2024, 1, 1, 10, 0, 0), 2.8],
        ["1", datetime(2024, 1, 1, 11, 0, 0), 4.0],
        ["1", datetime(2024, 1, 1, 12, 0, 0), 2.0],
        ["1", datetime(2024, 1, 1, 13, 0, 0), 53.9],
        ["1", datetime(2024, 1, 1, 14, 0, 0), 3.9],
        ["1", datetime(2024, 1, 1, 15, 0, 0), 2.9],
        ["1", datetime(2024, 1, 1, 16, 0, 0), 2.3],
        ["1", datetime(2024, 1, 1, 17, 0, 0), 4.3],
        ["1", datetime(2024, 1, 1, 18, 0, 0), 1.8],
        ["1", datetime(2024, 1, 1, 19, 0, 0), 2.5],
        ["1", datetime(2024, 1, 1, 20, 0, 0), 3.1],
        ["1", datetime(2024, 1, 1, 21, 0, 0), None],
        ["1", datetime(2024, 1, 1, 22, 0, 0), 2.2],
        ["1", datetime(2024, 1, 1, 23, 0, 0), 4.4],
    ]

    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assertDataFrameEqual(result_df, expected_df)


def test_turbine_sensors_data_imputation(spark, sample_data_with_issues):
    """Test the turbine_sensors_data_imputation function."""
    anomaly_detection_df = turbine_sensors_anomalies_detection(sample_data_with_issues)
    result_df = turbine_sensors_data_imputation(anomaly_detection_df)

    expected_schema = StructType(
        [
            StructField("turbine_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("power_output", DoubleType(), True),
            StructField("timestamp_in_sec", LongType(), True),
            StructField("power_output_mean", DoubleType(), True),
            StructField("power_output_stddev", DoubleType(), True),
            StructField("power_output_anomaly", BooleanType(), True),
            StructField("power_output_source_value", DoubleType(), True),
        ]
    )

    expected_data = [
        [
            "1",
            datetime(2024, 1, 1, 0, 0, 0),
            2.7,
            1704067200,
            5.222727272727273,
            10.909974982214859,
            False,
            2.7,
        ],
        [
            "1",
            datetime(2024, 1, 1, 1, 0, 0),
            4.4,
            1704070800,
            5.222727272727273,
            10.909974982214859,
            False,
            4.4,
        ],
        [
            "1",
            datetime(2024, 1, 1, 2, 0, 0),
            2.9,
            1704074400,
            5.222727272727273,
            10.909974982214859,
            False,
            2.9,
        ],
        [
            "1",
            datetime(2024, 1, 1, 3, 0, 0),
            1.8,
            1704078000,
            5.222727272727273,
            10.909974982214859,
            False,
            1.8,
        ],
        [
            "1",
            datetime(2024, 1, 1, 4, 0, 0),
            2.3,
            1704081600,
            5.222727272727273,
            10.909974982214859,
            False,
            2.3,
        ],
        [
            "1",
            datetime(2024, 1, 1, 5, 0, 0),
            2.2,
            1704085200,
            5.222727272727273,
            10.909974982214859,
            False,
            2.2,
        ],
        [
            "1",
            datetime(2024, 1, 1, 6, 0, 0),
            4.2,
            1704088800,
            5.222727272727273,
            10.909974982214859,
            False,
            4.2,
        ],
        ["1", datetime(2024, 1, 1, 7, 0, 0), 2.9, None, None, None, None, None],
        [
            "1",
            datetime(2024, 1, 1, 8, 0, 0),
            1.6,
            1704096000,
            5.222727272727273,
            10.909974982214859,
            False,
            1.6,
        ],
        [
            "1",
            datetime(2024, 1, 1, 9, 0, 0),
            2.7,
            1704099600,
            5.222727272727273,
            10.909974982214859,
            False,
            2.7,
        ],
        [
            "1",
            datetime(2024, 1, 1, 10, 0, 0),
            2.8,
            1704103200,
            5.222727272727273,
            10.909974982214859,
            False,
            2.8,
        ],
        [
            "1",
            datetime(2024, 1, 1, 11, 0, 0),
            4.0,
            1704106800,
            5.222727272727273,
            10.909974982214859,
            False,
            4.0,
        ],
        [
            "1",
            datetime(2024, 1, 1, 12, 0, 0),
            2.0,
            1704110400,
            5.222727272727273,
            10.909974982214859,
            False,
            2.0,
        ],
        [
            "1",
            datetime(2024, 1, 1, 13, 0, 0),
            2.95,
            1704114000,
            5.222727272727273,
            10.909974982214859,
            True,
            53.9,
        ],
        [
            "1",
            datetime(2024, 1, 1, 14, 0, 0),
            3.9,
            1704117600,
            5.222727272727273,
            10.909974982214859,
            False,
            3.9,
        ],
        [
            "1",
            datetime(2024, 1, 1, 15, 0, 0),
            2.9,
            1704121200,
            5.222727272727273,
            10.909974982214859,
            False,
            2.9,
        ],
        [
            "1",
            datetime(2024, 1, 1, 16, 0, 0),
            2.3,
            1704124800,
            5.222727272727273,
            10.909974982214859,
            False,
            2.3,
        ],
        [
            "1",
            datetime(2024, 1, 1, 17, 0, 0),
            4.3,
            1704128400,
            5.222727272727273,
            10.909974982214859,
            False,
            4.3,
        ],
        [
            "1",
            datetime(2024, 1, 1, 18, 0, 0),
            1.8,
            1704132000,
            5.222727272727273,
            10.909974982214859,
            False,
            1.8,
        ],
        [
            "1",
            datetime(2024, 1, 1, 19, 0, 0),
            2.5,
            1704135600,
            5.222727272727273,
            10.909974982214859,
            False,
            2.5,
        ],
        [
            "1",
            datetime(2024, 1, 1, 20, 0, 0),
            3.1,
            1704139200,
            5.222727272727273,
            10.909974982214859,
            False,
            3.1,
        ],
        ["1", datetime(2024, 1, 1, 21, 0, 0), 2.65, None, None, None, None, None],
        [
            "1",
            datetime(2024, 1, 1, 22, 0, 0),
            2.2,
            1704146400,
            5.222727272727273,
            10.909974982214859,
            False,
            2.2,
        ],
        [
            "1",
            datetime(2024, 1, 1, 23, 0, 0),
            4.4,
            1704150000,
            5.222727272727273,
            10.909974982214859,
            False,
            4.4,
        ],
    ]

    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assertDataFrameEqual(result_df, expected_df)
