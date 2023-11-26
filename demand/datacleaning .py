import pandas as pd

df = pd.read_csv(
    "algo-oil/demand/datademand/U.S._Exports_of_Finished_Petroleum_Products.csv"
)
df[["Month", "Year"]] = df["Month Year "].str.split(" ", expand=True)
print(df)

df.drop("Month Year ", axis=1, inplace=True)
df.to_csv("algo-oil/demand/datademand/cleaned_US_exports.csv", sep=",")
