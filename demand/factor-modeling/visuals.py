import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

us_exports_df = pd.read_csv("demand/cleaned-data/cleaned_US_exports.csv")
print(us_exports_df)

us_motor_df = pd.read_csv("demand/cleaned-data/cleaned_US_gasoline_production.csv")
print(us_motor_df)

merged_data = pd.merge(us_exports_df, us_motor_df, on="Year")

fig, ax1 = plt.subplots(figsize=(10, 6))


ax1.plot(
    merged_data["Year"],
    merged_data["U.S. Exports of Finished Petroleum Products Thousand Barrels per Day"],
    color="blue",
    label="US Exports",
)
ax1.set_xlabel("Year")
ax1.set_ylabel("U.S. Exports (Thousand Barrels per Day)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(
    merged_data["Year"],
    merged_data["U.S. Product Supplied of Finished Motor Gasoline (Thousand Barrels)"],
    color="red",
    label="U.S. Product Supplied of Finished Motor Gasoline (Thousand Barrels)",
)
ax2.set_ylabel(
    "U.S. Product Supplied of Finished Motor Gasoline (Thousand Barrels)", color="red"
)
ax2.tick_params(axis="y", labelcolor="red")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines = lines_1 + lines_2
labels = labels_1 + labels_2
ax1.legend(lines, labels, loc="upper left")

plt.title(
    "Comparison of US Exports and U.S. Product Supplied of Finished Motor Gasoline"
)
plt.show()
