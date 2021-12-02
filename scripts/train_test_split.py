import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("aud/aud_final_proj/csv/data.csv", header=None)

id = 0
seen = []

for idx, row in df.iterrows():
    pass