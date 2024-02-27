### Fake News Detectionv###
### Parameters: 
    # Title
    # Text
    # Subject
    # Date
####################################
    # Index #
# Importing Libraries
# Reading csv file
# Inserting the status column
# merging the columns
# Cleaning data
# Train test split
# Applying LogisticRegression
# Accuracy score : 98%    
#####################################
# Importing Libraries

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
###################### “axis 0” represents rows and “axis 1” represents columns. ##################
df_Merge= pd.concat([df_Fake,df_True], axis=0)

# Cleaning data
df = df_Merge.drop(["title", "subject","date"], axis = 1)

df.isnull().sum()

df = df.sample(frac = 1)

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

def wp(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


df["text"] = df["text"].apply(wp)
x = df["text"]
y = df["status"]

#Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#Applying LogisticRegression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(xv_train,y_train)

pred_lr=lr.predict(xv_test)
print(classification_report(y_test, pred_lr))

#Accuracy score : 98%
print(accuracy_score(y_test, pred_lr))

#print(df_Fake.head(2))
#print(df_True.head(2))

