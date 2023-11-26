import pandas as pd
from io import StringIO

<<<<<<< HEAD


df = pd.read_csv(
    "datademand/U.S._Exports_of_Finished_Petroleum_Products.csv"
)
=======
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

df = pd.read_csv("datademand/U.S._Exports_of_Finished_Petroleum_Products.csv")
df[["Month", "Year"]] = df["Month Year "].str.split(" ", expand=True)


df.drop("Month Year ", axis=1, inplace=True)
df = df[pd.to_numeric(df['Year'], errors= "coerce") >= 1997]
df["Month"] = df["Month"].map(months)
print(df)
df.to_csv("datademand/cleaned_US_exports.csv", sep=",", index=False)


cols = ["Letters", "Year", "Gallons per Vehicle", "Randint", "Desc", "Metric"]
motordf = pd.read_csv("datademand/MER_T01_08.csv", names=cols)
print(motordf["Year"].dtype)
motordf["Year"] = motordf["Year"] // 100

finalmotordf = motordf[["Year", "Gallons per Vehicle"]]
<<<<<<< HEAD
finalmotordf = finalmotordf[pd.to_numeric(finalmotordf['Year'], errors= "coerce") >= 1997]
print(finalmotordf)
finalmotordf.to_csv("datademand/cleaned_US_Fuel_Consumption.csv")


#cleaning data for monthly gasoline production 

gasoline_df = pd.read_csv("datademand/MGFUPUS1m.csv")

#Extract year and month
gasoline_df[['Month', 'Year']] = gasoline_df['Date'].str.split('-', expand=True)


# Reorganize columns
gasoline_df = gasoline_df[['Year', 'Month', 'U.S. Product Supplied of Finished Motor Gasoline (Thousand Barrels)']]
gasoline_df = gasoline_df[pd.to_numeric(gasoline_df['Year'], errors= "coerce") >= 1997]
gasoline_df.to_csv("datademand/cleaned_US_gasoline_production.csv")


print(gasoline_df)

#using EIA monthly energy report 
energy_df = pd.read_csv("datademand/1._U.S._Energy_Markets_Summary.csv")
energy_df = energy_df.transpose()
energy_df = energy_df[[9, 26, 27]]
energy_df.drop(["source key", "remove", "map", "linechart", "units", ], axis=0, inplace=True)
energy_df = energy_df.rename(columns = {9 : "U.S. Electricity Consumption (billion kilowatthours per day)", 26: 'U.S. Heating Degree Days (degree days)', 27: 'U.S. Cooling Degree Days (degree days)'})
energy_df.drop("Unnamed: 1", axis=0, inplace=True)
energy_df.to_csv('datademand/cleaned_US_electricity_consumption_HDDCDD')
print(energy_df)
=======
finalmotordf = finalmotordf[
    pd.to_numeric(finalmotordf["Year"], errors="coerce") >= 1981
]
print(finalmotordf)
finalmotordf.to_csv("datademand/cleaned_US_Fuel_Consumption.csv")
>>>>>>> 1bd6feea3e38340ddf134621280fe231f2227355
