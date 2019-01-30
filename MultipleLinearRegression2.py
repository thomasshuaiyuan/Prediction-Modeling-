#Multiple Linear Regression



#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
#importing the dataset
dataset=pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding the Dummy variable trap
x = x[:,1:]


#Splitting the data set into the Test set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test Set Results
y_pred=regressor.predict(x_test)

#Building the optimal mode using Backward Elimination
import statsmodels.formula.api as sm
x= np.append(arr = np.ones((50, 1)).astype(int), values=x, axis=1)


x_opt = x[:, [0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:, [0,1,4]]
regressor_OLS=sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

                         
