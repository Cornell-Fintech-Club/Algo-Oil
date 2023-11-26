import pandas as pd
import io as StringIO

df = pd.read_csv(
    "datademand/U.S._Exports_of_Finished_Petroleum_Products.csv"
)
df[["Month", "Year"]] = df["Month Year "].str.split(" ", expand=True)


df.drop("Month Year ", axis=1, inplace=True)
print(df)
df.to_csv("datademand/cleaned_US_exports.csv", sep=",")


cols = ['Letters', 'Year', 'Gallons per Vehicle', 'Randint', 'Desc', 'Metric' ]
motordf = pd.read_csv("datademand/MER_T01_08.csv", names = cols)
motordf['Year'] = motordf["Year"].str[:4]
finalmotordf = motordf[["Year", "Gallons per Vehicle"]]
print(finalmotordf)
finalmotordf.to_csv("datademand/cleaned_US_Fuel_Consumption.csv")

