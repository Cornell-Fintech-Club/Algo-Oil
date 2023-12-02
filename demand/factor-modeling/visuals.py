import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

# us_exports_df = pd.read_csv("demand/cleaned-data/cleaned_US_exports.csv")
# print(us_exports_df)

# us_motor_df = pd.read_csv("demand/cleaned-data/cleaned_US_gasoline_production.csv")
# print(us_motor_df)

# merged_data = pd.merge(us_exports_df, us_motor_df, on="Year")

# fig, ax1 = plt.subplots(figsize=(10, 6))


# ax1.plot(
#     merged_data["Year"],
#     merged_data["U.S. Exports of Finished Petroleum Products Thousand Barrels per Day"],
#     color="blue",
#     label="US Exports",
# )
# ax1.set_xlabel("Year")
# ax1.set_ylabel("U.S. Exports (Thousand Barrels per Day)", color="blue")
# ax1.tick_params(axis="y", labelcolor="blue")
# ax1.grid(True)

# ax2 = ax1.twinx()
# ax2.plot(
#     merged_data["Year"],
#     merged_data["U.S. Product Supplied of Finished Motor Gasoline (Thousand Barrels)"],
#     color="red",
#     label="U.S. Product Supplied of Finished Motor Gasoline (Thousand Barrels)",
# )
# ax2.set_ylabel(
#     "U.S. Product Supplied of Finished Motor Gasoline (Thousand Barrels)", color="red"
# )
# ax2.tick_params(axis="y", labelcolor="red")

# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# lines = lines_1 + lines_2
# labels = labels_1 + labels_2
# ax1.legend(lines, labels, loc="upper left")

# plt.title(
#     "Comparison of US Exports and U.S. Product Supplied of Finished Motor Gasoline"
# )
# plt.show()


#graph the actual crude oil production for August 2023 to May 2015 against the prediction model 
prediction_df = pd.read_csv("demand/outputs/best_loss_data.csv")
actual_crude_prod = pd.read_csv("demand/cleaned-data/test.csv")
prediction_df.drop("Unnamed: 0", axis = 1, inplace = True)
prediction_df.columns = ['Predictions']


#reverse the order so its not reverse chronological order 
newpred = prediction_df[::-1].reset_index(drop=True)
# Reverse the order of rows in df2
newact = actual_crude_prod[::-1].reset_index(drop=True)


# Creating a date range with month frequency
date_range = pd.date_range(start='2015-05-01', end='2023-08-01', freq='MS')


print(prediction_df)
plt.figure(figsize=(13, 5.7))
plt.plot(newpred['Predictions'], label='Predictions')
plt.plot(newact['U.S. Crude Oil Production (million barrels per day)'], label=' Actual U.S. Crude Oil Production')


plt.xticks(range(0, len(newact.index), 7), date_range[::7].strftime('%b %Y'), rotation=45)

plt.xlabel('Month and Day Data Points')
plt.ylabel('U.S Crude Oil Prod (million barrels per day)')
plt.legend()
# plt.xticks(date_range, rotation=45)
plt.show()
# plt.tight_layout()
plt.show