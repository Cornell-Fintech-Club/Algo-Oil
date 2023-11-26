import pandas as pd
from io import StringIO

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

df["Month"] = df["Month"].map(months)
print(df)
df.to_csv("datademand/cleaned_US_exports.csv", sep=",", index=False)


cols = ["Letters", "Year", "Gallons per Vehicle", "Randint", "Desc", "Metric"]
motordf = pd.read_csv("datademand/MER_T01_08.csv", names=cols)
print(motordf["Year"].dtype)
motordf["Year"] = motordf["Year"] // 100

finalmotordf = motordf[["Year", "Gallons per Vehicle"]]
finalmotordf = finalmotordf[
    pd.to_numeric(finalmotordf["Year"], errors="coerce") >= 1981
]
print(finalmotordf)
finalmotordf.to_csv("datademand/cleaned_US_Fuel_Consumption.csv")
