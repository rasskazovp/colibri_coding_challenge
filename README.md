# Introduction
Coding challenge solution provided by Pavlo Rasskazov
# Data and Processing
### Assumptions and observations
- Wind speed is expected to be a non-negative value.
- Wind direction is expected to be between 0 and 359 degrees.
- Power output is expected to be a non-negative value.
- Power output reading is considered to be an anomaly if it deviates over 2 standard deviations from the mean over the past 24 hours.
- Hourly granularity and daily extraction frequency.

### Decisions
- Ingestion to bronze, data type conversion, and data quality validation will be performed in a streaming manner.
- Silver and gold layer processing will be performed in batches to utilize more complex anomaly detection and imputation logic.
- `power_output` will be flaged as anomaly if it diviates for more then 2 standard deviations from average in range between -24 hours and +24 hours.
- If a sensor reading provides invalid data or if a sensor reading is missing, then values will be imputed by finding an average between preceding and following readings for the same turbine. If a record doesn't have a next or previous value, then the imputed value will be set to the previous or next value (whichever is available).
- Original `power_output` values will be maintained for traceability.
- Sensor readings will be summarized on a daily level. The possibility for weekly and monthly summarization will also be implemented.

# Design decisions

### Medalion Architecture
This project follows the Medallion Architecture to structure data processing into distinct layers, ensuring a clear separation of concerns:

Bronze – Raw, incremental data ingestion with append-only operations (no modifications).
Silver – Cleaned and curated data with necessary transformations and validations.
Gold – Aggregated and summarized data, optimized for reporting and analytics.

### Data Live Tables
This project utilizes Databricks Delta Live Tables (DLT) to process and orchestrate turbine sensor data efficiently.

The core processing logic is implemented in Python, ensuring flexibility for future adaptations. This approach allows for an easy transition to Databricks Jobs/Notebooks. Additionally, it facilitates potential migration to alternative platforms if needed.

### Autoloader and Streaming
Data ingestion into the bronze layer is implemented using Autoloader, ensuring that only new files from the source folder are processed. Additionally, Streaming Tables are used for data type conversion and data quality validation, enabling these transformations to be applied only to newly ingested records.

### Unity Catalog
A crucial component of a modern data lakehouse, providing unified governance and secure data sharing across teams.

### Databricks Asset Bindles (DAB)
This project is built using Databricks Asset Bundles (DAB), which simplifies code deployment and enhances development workflow when working with Databricks from an IDE.

# Getting Started

### Configure your IDE environment
1. Project connfigured with `poetry`, so you can easealy create your virtual environment and manage dependencies with `poetry`.
2. Install pre-commit hooks by running `pre-commit install`

### Workspace Requirments
To deploy and run this project, the Databricks workspace must meet the following requirements:

 - Databricks Premium Workspace
 - Unity Catalog Integration – The workspace must be connected to Unity Catalog, with a catalog named `colibri` created and ready for use.
 - Serverless Compute Enabled
 - Landing Location Configured as External Storage or Volume
 - Databricks CLI installed (latest version recommended)

### Deployment steps
1. Set your Databricks Id in `databricks.yml` file.
2. Create profile to connect your Databricks workspace (if not exists) `databricks auth login`
2. Execute command `databricks bundle deploy --var="landing_root_path=<LANDING_PATH>" -p <YOUR_PROFILE_NAME>`

### Execute tests
Run `pytest` in yout IDE to execute defined unit tests.


# Extras
- poetry
- pytest
- pre-commit
- ci/cd
    - unit-test
    - trufflehog
    - bandit
    - ruff
    - dab-validate
    - multi-stage deployment