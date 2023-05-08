import os
import re
import pandas as pd

data_format = pd.read_excel("Project/2/data/2 PROBE/aimms_data_format.xlsx")
columns = [str.strip(";") for str in data_format.iloc[0].tolist()]

with open("Project/2/data/2 PROBE/data.csv", "w") as f:
    f.write(",".join(columns) + "\n")
    data_path = "Project/2/data/2 PROBE/160909_B_LVB"
    for file in sorted(os.listdir(data_path)):
        with open(f"{data_path}/{file}") as g:
            for content in g.readlines():
                content = re.sub(r"^\s+", "", content)
                content = re.sub(r"\s+$", "", content)
                content = re.sub(r"\s+", ",", content)
                f.write(content + "\n")
