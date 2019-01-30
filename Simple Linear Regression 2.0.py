#Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
#importing the dataset
dataset=pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values


#Splitting the data set into the Test set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_testA = train_test_split(x, y, test_size=0.2, random_state=0)




