import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

# Reading csv file

df_Fake= pd.read_csv('Fake.csv')
df_True= pd.read_csv('True.csv')

# Inserting the status column
df_Fake['status']=0; 
df_True['status'] =1

# merging the columns
df_Merge= pd.concat([df_Fake,df_True], axis=0)

# Cleaning data
df = df_Merge.drop(["title", "subject","date"], axis = 1)
