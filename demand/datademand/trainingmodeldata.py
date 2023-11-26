import pandas as pd 

us_exports_df = pd.read_csv("cleaned_US_exports.csv")
print(us_exports_df)

us_motor_df = pd.read_csv("cleaned_US_gasoline_production.csv")
print(us_motor_df)

us_energy_weather = pd.read_csv('cleaned_US_electricity_consumption_HDDCDD')

merged_data = pd.merge(us_exports_df, us_motor_df, on=['Year', 'Month'])
merged_data = pd.merge(merged_data, us_energy_weather, on=['Year', 'Month'])
merged_data.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis = 1, inplace= True )


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

merged_data.to_csv('cleaned_final_TEST.csv')
print(merged_data)