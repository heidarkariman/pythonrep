import numpy
from scipy import stats
import matplotlib.pyplot as plt

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.mean(speed)

#print(x)

x = numpy.median(speed)

#print(x)

x = stats.mode(speed, keepdims=False)

#print(x)

# Standard Deviation is often represented by the symbol Sigma: σ
# Variance is often represented by the symbol Sigma Squared: σ2

#deviation
σ = numpy.std(speed)

#print(σ)

#variance
σ2 = numpy.var(speed)

#print(σ2)

#percentile
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

x = numpy.percentile(ages, 50)

#print(x)

#random data produce 
#Create an array containing 250 random floats between 0 and 5:

x = numpy.random.uniform(0.0, 5.0, 1000000)

# plt.hist(x, 100)
# plt.show()


x = numpy.random.normal(5.0, 1.0, 100000)

# plt.hist(x, 100)
# plt.show()

#linear regression
# 
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]


slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

#print(r)
speed = myfunc(10)
#print(speed)


# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()

#Polynomial Regression

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

from sklearn.metrics import r2_score
#print(r2_score(y, mymodel(x)))

speed = mymodel(3.5)
#print(speed)

# plt.scatter(x, y)
# plt.plot(myline, mymodel(myline))
# plt.show()

#mulity parametric analyse

import pandas

df = pandas.read_csv("data.csv")
X = df[['Weight', 'Volume']]
y = df['CO2']

from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

#print(predictedCO2)
#print(regr.coef_)


#very important StandardScaler to bring all variant parameters in a compairable scaler

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])
#print(predictedCO2)

#Train/Test

import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(test_y, mymodel(test_x))

#print(r2)
#print(mymodel(5))


#decision tree

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("data2.csv")

# map categorical features to numerical values
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)

d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

# define the features and target variable
features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

# fit the decision tree classifier
dtree = DecisionTreeClassifier()
dtree.fit(X, y)

print(dtree.predict([[40, 10, 7, 1]]))
print(dtree.predict([[40, 10, 6, 1]]))

# plot the decision tree
plt.figure(figsize=(10, 6))
tree.plot_tree(dtree, feature_names=features, filled=True, rounded=True)
#plt.show()

# confusion matrix



import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()