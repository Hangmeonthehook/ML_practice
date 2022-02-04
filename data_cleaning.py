import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# load dataset
os.chdir('C:\\Users\\hq217\\OneDrive\\Desktop\\Machine Learning A-Z Chinese Template Folder\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Data_Preprocessing')
dataset = pd.read_csv('Data.csv')

# independent variable
X = dataset.iloc[: , 0:3]
X

# dependent variable
y = dataset.iloc[:,-1]
y

from sklearn.impute import SimpleImputer
# fill na value with mean
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
# label variable with the right type
# country to dummy variable
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

# change Yes/No to number
# doesn't need onehotencoder
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y)

# split data into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
# standardisation
# noramlisation
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
