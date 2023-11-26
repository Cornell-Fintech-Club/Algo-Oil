import pandas as pd
from io import StringIO

df = pd.read_csv(
    "datademand/U.S._Exports_of_Finished_Petroleum_Products.csv"
)
df[["Month", "Year"]] = df["Month Year "].str.split(" ", expand=True)



df.drop("Month Year ", axis=1, inplace=True)
print(df)
df.to_csv("datademand/cleaned_US_exports.csv", sep=",")


cols = ['Letters', 'Year', 'Gallons per Vehicle', 'Randint', 'Desc', 'Metric' ]
motordf = pd.read_csv("datademand/MER_T01_08.csv", names = cols)
print(motordf["Year"].dtype)
motordf['Year'] = motordf["Year"]//100

finalmotordf = motordf[["Year", "Gallons per Vehicle"]]
finalmotordf = finalmotordf[pd.to_numeric(finalmotordf['Year'], errors= "coerce") >= 1981]
print(finalmotordf)
finalmotordf.to_csv("datademand/cleaned_US_Fuel_Consumption.csv")

