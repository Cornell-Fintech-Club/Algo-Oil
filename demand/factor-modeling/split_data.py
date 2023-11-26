import pandas as pd

val_size, test_size = 100, 100
df = pd.read_csv("../datademand/cleaned_final_TEST.csv")
test = df.head(test_size)
val = df.iloc[test_size : test_size + val_size]
train = df.iloc[test_size + val_size :]
test.to_csv("../datademand/test.csv")
val.to_csv("../datademand/val.csv")
train.to_csv("../datademand/train.csv")
