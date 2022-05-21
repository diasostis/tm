#import dependancies modules
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#read excel file as Pandas DataFrame
df = pd.read_excel("Project38Titanic.xlsx")
print(type(df))
print(df)

#open excel workbook
wb = openpyxl.load_workbook("Project38Titanic.xlsx")
print(type(wb))
print(wb)

#worksheet object
ws = wb[wb.sheetnames[0]]
print(type(ws))
print(ws)

#indices corresponding to all hidden colums
hidden_cols_idx = [
    string.ascii_uppercase.index(col_name)
    for col_name in [
        col
        for col, dimension in ws.column_dimensions.items()
        if dimension.hidden
    ]
]

#find names of colums corresponding to hidden column indices
hidden_cols_name = df.columns[hidden_cols_idx].tolist()

#drop hidden columns and rows
df.drop(hidden_cols_name, axis=1, inplace=True)
df.drop(hidden_cols_idx, axis=0 , inplace=True)

#reset index
df.reset_index(drop=True, inplace=True)

df
