import pandas as pd

df = pd.read_csv("../data/1 QAR/CX254_2022070122_1-results.csv")

columns = "AIRSPEED,WSPD,WDIR,VWRZ,EDR,RUNTIME".split(",")
df = df[columns]

df.to_csv("../data/1 QAR/data.csv", index=False)
