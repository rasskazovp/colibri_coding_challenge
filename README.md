# Introduction
Coding challenge solution provided by Pavlo Rasskazov
# Data and Processing
### Assumptions and observations
- wind_speed expected to be not negative value
- wind_direction expected to be between 0 and 359 degrees
- power_output expected to be not negative value
- power_output reading considered to be anomaly if it deviats over 2 stddev from mean over past 24h
- hourly granularity and daily extraction freequency

### Decisions
- Despite it make sense to process sensor data in streaming fasion we will process silver and gold layer using batches to utilise more complex data quality handling.
- If sensor reading provides invalid data or if sensor reading is missing, then values should be imputed by finding avarage between preceeding and following reading.
- All imputed values shoudl be flagged and original values mainteined for tracability
- Verification of data validity at the start and begining of daily extracted period can be inprecise because of lack of enough preceeding or following data. So each run will processing new data plus previous day, to source historical readings and update it if neaded.
# Design decisions
- DLT
- UC
- DAB
- AutoLoader
- 
# Getting Started

# Extras
- poetry
- pytest
- pre-commit
- ci/cd