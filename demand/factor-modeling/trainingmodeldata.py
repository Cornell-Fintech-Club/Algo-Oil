"""
Generating the data for testing the model.

Combines the US Exports, US Motors, and US Energy Weather dataframes
"""

# Imports
import pandas as pd

# Setting up data
## U.S. Exports Dataframe
us_exports_df = pd.read_csv("demand/cleaned-data/cleaned_US_exports.csv")
print(us_exports_df)

## U.S. Cleaned Gasoline Production
us_motor_df = pd.read_csv("demand/cleaned-data/cleaned_US_gasoline_production.csv")
print(us_motor_df)

## US Weather
us_energy_weather = pd.read_csv(
    "demand/cleaned-data/cleaned_US_electricity_consumption_HDDCDD_crudeprod"
)
print(us_energy_weather)

## Joining all the data
merged_data = pd.merge(us_exports_df, us_motor_df, on=["Year", "Month"])
merged_data = pd.merge(merged_data, us_energy_weather, on=["Year", "Month"])
merged_data.drop(["Unnamed: 0_x", "Unnamed: 0_y"], axis=1, inplace=True)

## Converting the months of the dataframe to numerical values
months = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}

merged_data["Month"] = merged_data["Month"].map(months)

# Saving the final results of the joined data
merged_data.to_csv("demand/cleaned-data/cleaned_final_TEST.csv")
print(merged_data)
