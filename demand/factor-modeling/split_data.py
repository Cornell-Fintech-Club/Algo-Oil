import pandas as pd

val_size, test_size = 100, 100
df = pd.read_csv("demand/cleaned-data/cleaned_final_TEST.csv")
test = df.head(test_size)
val = df.iloc[test_size : test_size + val_size]
train = df.iloc[test_size + val_size :]
test.to_csv("demand/cleaned-data/test.csv")
val.to_csv("demand/cleaned-data/val.csv")
train.to_csv("demand/cleaned-data/train.csv")
