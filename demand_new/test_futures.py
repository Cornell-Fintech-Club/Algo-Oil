from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


X = pd.read_csv("USexportsofcrude.csv")
population = pd.read_csv("uspopulation.csv")
electricity = pd.read_csv("electricity.csv")

X["Date"] = pd.to_datetime(X["Date"], format="%b %d, %Y")
X["Year"] = X["Date"].dt.year
X["Month"] = X["Date"].dt.month
X = (
    X.groupby(["Year", "Month"])[
        "Weekly U.S. Exports of Crude Oil  (Thousand Barrels per Day)"
    ]
    .mean()
    .reset_index()
)
X.rename(
    columns={
        "Weekly U.S. Exports of Crude Oil  (Thousand Barrels per Day)": "Monthly Average Exports (Thousand Barrels per Day)"
    },
    inplace=True,
)
X["Value"] = population["Value"]
# X["Production"] = electricity["production"]
X["Electricity"] = electricity["electricity"]
X["Heating"] = electricity["heating"]
X["Cooling"] = electricity["cooling"]


X["Date"] = pd.to_datetime(
    X["Year"].astype(int).astype(str)
    + "-"
    + X["Month"].astype(int).apply(lambda x: f"{x:02d}")
)
date = pd.DataFrame(X["Date"])
X = X.iloc[:-1]
print(X)

print(date)
del X["Date"]

# date = X["Date"]
# date = pd.to_datetime(date)
# print(date)


# X['YearMonth'] = X['Date'].dt.to_period('M')
# exportscrude = X.groupby('YearMonth')['Weekly U.S. Exports of Crude Oil  (Thousand Barrels per Day)'].mean().reset_index()
# exportscrude['YearMonth'] = exportscrude['YearMonth'].dt.to_timestamp()
# print(exportscrude)

# X = X.transpose()
# X.columns = X.iloc[0]
# X = X.iloc[1:]
# X = X.reset_index()


Y = pd.read_csv("finishedmotorgasoline.csv")
Y = pd.DataFrame(Y)
Y = Y.drop("Date", axis=1)

Y.to_csv("Y_data.csv")
date.to_csv("Dates.csv")
Y = Y.iloc[1:]
Y = Y.reset_index()
Y = Y.drop("index", axis=1)

print(Y)

date = date.iloc[1:]
date = date.reset_index()
date = date.drop("index", axis=1)

# Y[["Month", "Year"]] = Y["Month"].str.split(" ", expand=True)
# Y["Year"] = Y["Year"].astype(int)
# Y = (
#     Y.groupby("Year")["U.S. Field Production of Crude Oil Thousand Barrels"]
#     .sum()
#     .reset_index()
# )

# del Y["Year"]
# print(Y)

m, n = X.shape

# # test set is 10%
# # train set is 90%
size = 0.95
X_train = X.loc[: np.floor(m * size)]
X_test = X.loc[np.floor(m * size) + 1 :]

Y_train = Y.loc[: np.floor(m * size)]
Y_test = Y.loc[np.floor(m * size) + 1 :]

date_train = date.loc[: np.floor(m * size)]
date_test = date.loc[np.floor(m * size) + 1 :]
print(len(X_train))
print(len(Y_train))

# print(X_test)
# print(X_train)
# print(Y_test)
# print(Y_train)


lr = LinearRegression()

# Train the model using the training sets
lr.fit(X_train, Y_train)
predict_train = lr.predict(X_train)
predict_test = lr.predict(X_test)
Y_predict = np.concatenate((predict_train, predict_test), axis=0)

# The coefficients
print("Coefficients: \n", lr.coef_)
# The mean square error
print("Residual sum of squares: %.2f" % np.mean((Y_predict - Y) ** 2))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % lr.score(X, Y))

plt.xticks(rotation=45)
plt.plot_date(
    date,
    Y,
    fmt="ko-",
    xdate=True,
    ydate=False,
    label="Real value",
    ms=3,
)

print(type(lr.predict(X_train)))
plt.plot_date(
    date_train,
    predict_train,
    fmt="o-",
    xdate=True,
    ydate=False,
    label="Predicted train",
    color="#0074D9",
    ms=3,
)

plt.plot_date(
    date_test,
    predict_test,
    fmt="o-",
    xdate=True,
    ydate=False,
    label="Predicted test",
    color="#228B22",
    ms=3,
)


plt.legend(loc="upper center")
plt.ylabel("U.S. Demand of Crude Oil")
plt.title("Predicted US Demand of Crude Oil Modeled by Gasoline Supply")
plt.grid()
plt.show()


################################################

# futures = pd.read_csv("correlation analysis/future_combined.csv")
# futures = futures.reset_index()

# futures["Date"] = pd.to_datetime(futures["Date"], format="%d-%m-%Y")
# future_date = futures.loc[:, ["Date"]]
# futures["Date2num"] = futures["Date"].apply(lambda x: mdates.date2num(x))
# del futures["Date"]
# del futures["index"]
# # print(futures)


# lr.predict(futures)

# plt.xticks(rotation=45)
# plt.plot_date(
#     date,
#     Y,
#     fmt="ko-",
#     xdate=True,
#     ydate=False,
#     label="Real value",
#     ms=3,
# )

# # print(type(lr.predict(X_train)))
# plt.plot_date(
#     date,
#     Y_predict,
#     fmt="o-",
#     xdate=True,
#     ydate=False,
#     label="Predicted value",
#     color="#0074D9",
#     ms=3,
# )

# plt.plot_date(
#     future_date,
#     lr.predict(futures),
#     fmt="o-",
#     xdate=True,
#     ydate=False,
#     label="Future value",
#     color="#FF4136",
#     ms=3,
# )

# plt.legend(loc="upper center")
# plt.ylabel("Close prices")
# plt.title("Relative Frequency of Searches for Gemini Model")
# plt.grid()
# plt.show()


# Coefficients:
#  [[  273.34822785  -142.82185374  -640.71173023 -1268.02892289]]
# Residual sum of squares: 47622848828.73
# Variance score: 0.89
