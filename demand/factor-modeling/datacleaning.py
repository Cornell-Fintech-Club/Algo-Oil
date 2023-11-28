import pandas as pd
from io import StringIO


df = pd.read_csv("demand/initial-data/U.S._Exports_of_Finished_Petroleum_Products.csv")


df = pd.read_csv("demand/initial-data/U.S._Exports_of_Finished_Petroleum_Products.csv")
df[["Month", "Year"]] = df["Month Year "].str.split(" ", expand=True)


df.drop("Month Year ", axis=1, inplace=True)
df = df[pd.to_numeric(df["Year"], errors="coerce") >= 1997]

print(df)
df.to_csv("demand/cleaned-data/cleaned_US_exports.csv", sep=",", index=False)


cols = ["Letters", "Year", "Gallons per Vehicle", "Randint", "Desc", "Metric"]
motordf = pd.read_csv("demand/initial-data/MER_T01_08.csv", names=cols)
print(motordf["Year"].dtype)
motordf["Year"] = motordf["Year"] // 100

finalmotordf = motordf[["Year", "Gallons per Vehicle"]]
finalmotordf = finalmotordf[
    pd.to_numeric(finalmotordf["Year"], errors="coerce") >= 1997
]
print(finalmotordf)
finalmotordf.to_csv("demand/cleaned-data/cleaned_US_Fuel_Consumption.csv")


# cleaning data for monthly gasoline production

gasoline_df = pd.read_csv("demand/initial-data/MGFUPUS1m.csv")

# Extract year and month
gasoline_df[["Month", "Year"]] = gasoline_df["Date"].str.split("-", expand=True)


# Reorganize columns
gasoline_df = gasoline_df[
    [
        "Year",
        "Month",
        "U.S. Product Supplied of Finished Motor Gasoline (Thousand Barrels)",
    ]
]
gasoline_df = gasoline_df[pd.to_numeric(gasoline_df["Year"], errors="coerce") >= 1997]
gasoline_df.to_csv("demand/cleaned-data/cleaned_US_gasoline_production.csv")


print(gasoline_df)

# using EIA monthly energy report
energy_df = pd.read_csv(
    "demand/initial-data/1._U.S._Energy_Markets_Summary.csv", header=None
)
energy_df = energy_df.transpose()
energy_df = energy_df[[0, 3, 10, 27, 28]]
energy_df.drop([0, 1, 2, 3, 4], axis=0, inplace=True)
energy_df = energy_df.rename(
    columns={
        0: "Date",
        3: "U.S. Crude Oil Production (million barrels per day)",
        10: "U.S. Electricity Consumption (billion kilowatthours per day)",
        27: "U.S. Heating Degree Days (degree days)",
        28: "U.S. Cooling Degree Days (degree days)",
    }
)
energy_df.drop(5, axis=0, inplace=True)
energy_df[["Month", "Year"]] = energy_df["Date"].str.split(" ", expand=True)
energy_df.drop("Date", axis=1, inplace=True)
energy_df.to_csv(
    "demand/cleaned-data/cleaned_US_electricity_consumption_HDDCDD_crudeprod"
)
print(energy_df)
