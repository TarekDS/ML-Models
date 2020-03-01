# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:34:28 2019

@author: T
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR
from copy import deepcopy


#importing Arrhythmia data
url= ("https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data")
df = pd.read_csv(url, header = None)

df.head()
df.describe()
df.dtypes

#verifying target label
df[279].value_counts()

#removing range to columns
df.drop(df.iloc[:, 16:279], inplace=True, axis=1)

# deleting another obsolete column 
df = df.drop(15, axis=1)
df = df.drop(13, axis=1)


#adding column names. Last column in the data is the  Target Label
#df.columns = ['Age','Sex','Height','Weight','QRS','P-R','Q-T','Tinterval','Pinterval','QRS','P','QRST','J','Heartrate', 'Label']

#replacing "?" values with NaN
df = df.replace('?', np.NaN)
df.replace(' ', np.nan, inplace=True)

#converting muti-class Label into a binary class target label
arrhythmiadetection = df.loc[:, 279] > 1
#replacing the above values with median
df.loc[arrhythmiadetection, 279] = 0
#testing conversion
df[279].value_counts()


#Normalizing Dataset
from sklearn import preprocessing

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

#testing Normalized values
plt.hist(df.loc[:, 13])

#dropping rows with Nan but the code didnot drop nan
df.dropna(how='any').shape
df.fillna(0)
df.isnull().sum()

#unable to replce Nan values completely after using Dropna before and after normalization
#for that reason exporting the file to remove those values externally 
df.to_excel(r'C:\Users\T\Desktop\Python\Data science Uwash\finalproject.xlsx')

#importing the df again after removing null values 
datasource = (r'C:\Users\T\Desktop\Python\Data science Uwash\finalproject.csv')
df = pd.read_csv(datasource, header = None)

#spliting Feature and target label
from sklearn.model_selection import train_test_split
X = df.iloc[:,0:13].values
y = df.iloc[:,14].values
#Spliting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 2, test_size=0.30)

pd.DataFrame(y_train)


# Applying kNN regression 
# Apply the Model and saving results in KNNresults variable
print ("\n\nK Nearest Neighbors Regression\n")
k = 10 # number of neighbors to be used
distance_metric = 'euclidean'
regr = KNeighborsRegressor(n_neighbors=k, metric=distance_metric)
regr.fit(X_train, y_train)
print ("predictions for test set:")
print (regr.predict(X_test))
KNNresult = (regr.predict(X_test))
print ('actual target values:')
print (y_test)

# Applying SVM
# Apply the Model and saving results in SVMresults variable
print ("\n\nSupport Vector Machine Regression\n")
t = 0.001 # tolerance parameter
kp = 'rbf'
regr = SVR(kernel=kp, tol=t)
regr.fit(X_train, y_train)
print ("predictions for test set:")
print (regr.predict(X_test))
SVMresult = (regr.predict(X_test))
print ('actual target values:')
print (y_test)

# Logistic regression classifier
print ('\n\n\nLogistic regression classifier\n')
C_parameter = 50. / len(X_train) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter
#####################

#Training the Model
clf = LogisticRegression(C=C_parameter, multi_class=class_parameter, penalty=penalty_parameter, solver=solver_parameter, tol=tolerance_parameter)
clf.fit(X_train, y_train) 
print ('coefficients:')
print (clf.coef_) # each row of this matrix corresponds to each one of the classes of the dataset
print ('intercept:')
print (clf.intercept_) # each element of this vector corresponds to each one of the classes of the dataset

# Apply the Model and saving results in LRresults variable
print ('predictions for test set:')
print (clf.predict(X_test))
LRresults = (clf.predict(X_test))
print ('actual class values:')
print (y_test)

# Naive Bayes classifier
print ('\n\nNaive Bayes classifier\n')
nbc = GaussianNB() # default parameters are fine
nbc.fit(X_train, y_train)
print ("predictions for test set:")
print (nbc.predict(X_test))
NBCresult = (nbc.predict(X_test))
print ('actual class values:')
print (y_test)
####################

# Decision Tree classifier
print ('\n\nDecision Tree classifier\n')
clf = DecisionTreeClassifier() # default parameters are fine
clf.fit(X_train, y_train)
print ("predictions for test set:")
print (clf.predict(X_test))
DTCresult = (clf.predict(X_test))
print ('actual class values:')
print (y_test)
####################

# Random Forest classifier
estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
print ('\n\nRandom Forest classifier\n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
clf.fit(X_train, y_train)
print ("predictions for test set:")
print (clf.predict(X_test))
RFCresults = (clf.predict(X_test))
print ('actual class values:')
print (y_test)
####################

# k Nearest Neighbors classifier
print ('\n\nK nearest neighbors classifier\n')
k = 5 # number of neighbors
distance_metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
knn.fit(X_train, y_train)
print ("predictions for test set:")
print (knn.predict(X_test))
NNCresult = (knn.predict(X_test))
print ('actual class values:')
print (y_test)
###################

# Linear regression 
print ("\n\n\nBasic Linear Regression\n")
regr = LinearRegression()
regr.fit(X_train, y_train)
print ("Coefficients:")
print (regr.coef_)
print ("Intercept:")
print (regr.intercept_)
print ("predictions for test set:")
print (regr.predict(X_test))
LinearrResult = (regr.predict(X_test))
print ('actual target values:')
print (y_test)
#####################


# Ridge regression
print ("\n\nRidge Regression\n")
a = 0.5 # alpha parameter for regularization
t = 0.001 # tolerance parameter
regr = Ridge(alpha=a, tol=t)
regr.fit(X_train, y_train)
print ("Coefficients:")
print (regr.coef_)
print ("Intercept:")
print (regr.intercept_)
print ("predictions for test set:")
print (regr.predict(X_test))
RidgeRresult = (regr.predict(X_test))
print ('actual target values:')
print (y_test)
########################

# Lasso regression
print ("\n\nLasso Regression\n")
a = 0.5 # alpha parameter for regularization
t = 0.0001 # tolerance parameter
regr = Lasso(alpha=a, tol=t)
regr.fit(X_train, y_train)
print ("Coefficients:")
print (regr.coef_)
print ("Intercept:")
print (regr.intercept_)
print ("predictions for test set:")
print (regr.predict(X_test))
LassoRresult = (regr.predict(X_test))
print ('actual target values:')
print (y_test)
########################

# Support vector machine classifier
t = 0.001 # tolerance parameter
kp = 'rbf' # kernel parameter
print ('\n\nSupport Vector Machine classifier\n')
clf = SVC(kernel=kp, tol=t)
clf.fit(X_train, y_train)
print ("predictions for test set:")
print (clf.predict(X_test))
SVMCresult = (clf.predict(X_test))
print ('actual class values:')
print (y_test)
####################


#Calculating variance (not a part of assignemnt just fact checking)
ssr = np.sum((y_test - KNNresult)**2)
print(ssr)

ssr = np.sum((y_test - SVMresult)**2)
print(ssr)

ssr = np.sum((y_test - LRresults)**2)
print(ssr)

ssr = np.sum((y_test - NBCresult)**2)
print(ssr)

ssr = np.sum((y_test - DTCresult)**2)
print(ssr)

ssr = np.sum((y_test - RFCresults)**2)
print(ssr)

ssr = np.sum((y_test - NNCresult)**2)
print(ssr)

ssr = np.sum((y_test - LinearrResult)**2)
print(ssr)

ssr = np.sum((y_test - RidgeRresult)**2)
print(ssr)

ssr = np.sum((y_test - LassoRresult)**2)
print(ssr)

# Measuring KNN performance
# data to work with (T = target labels, y = prob. output of classifier, pred = closest class predictions)
T = y_test
y = KNNresult
Y = np.round(y, 0)

import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt

# Confusion Matrix
CM = confusion_matrix(T, Y)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, Y)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, Y)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, Y)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, Y)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")

#measuring SVM performance
# data to work with (T = target labels, y = prob. output of classifier, pred = closest class predictions)
T = y_test
y = SVMresult
Y = np.round(y, 0)

import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt

# Confusion Matrix
CM = confusion_matrix(T, Y)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, Y)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, Y)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, Y)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, Y)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")


#Measuring Linear Regression performance
# data to work with (T = target labels, y = prob. output of classifier, pred = closest class predictions)
T = y_test
y = LRresults
Y = np.round(y, 0)

import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt

# Confusion Matrix
CM = confusion_matrix(T, Y)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, Y)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, Y)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, Y)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, Y)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")

#Measuring NBC performance
# data to work with (T = target labels, y = prob. output of classifier, pred = closest class predictions)
T = y_test
y = NBCresult
Y = np.round(y, 0)

# Confusion Matrix
CM = confusion_matrix(T, Y)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, Y)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, Y)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, Y)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, Y)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")

#Measuring DTC performance
# data to work with (T = target labels, y = prob. output of classifier, pred = closest class predictions)
T = y_test
y = DTCresult
Y = np.round(y, 0)

# Confusion Matrix
CM = confusion_matrix(T, Y)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, Y)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, Y)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, Y)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, Y)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")

#Measuring RFC performance
# data to work with (T = target labels, y = prob. output of classifier, pred = closest class predictions)
T = y_test
y = RFCresults
Y = np.round(y, 0)

# Confusion Matrix
CM = confusion_matrix(T, Y)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, Y)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, Y)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, Y)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, Y)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")

#Measuring NNC performance
# data to work with (T = target labels, y = prob. output of classifier, pred = closest class predictions)
T = y_test
y = NNCresult
Y = np.round(y, 0)

# Confusion Matrix
CM = confusion_matrix(T, Y)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, Y)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, Y)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, Y)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, Y)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")

T = y_test
y = LRresults
Y = np.round(y, 0)

import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt

# Confusion Matrix
CM = confusion_matrix(T, Y)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, Y)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, Y)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, Y)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, Y)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(T, y) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, y), 2), "\n")

"""
# following are the comparison between the accuracy rate of all 3 models:
SVM                             Accuracy rate:    0.6397058823529411
Logistic regression             Accuracy rate: 0.625
KNN                             Accuracy rate: 0.6544117647058824

it is determined that SVM was the closest when predicting the class i.e. when it was 1 the person was normal and when it was 0 the person had a type of arrhythmia 

SVM                             Precision: 0.68                   Recall: 0.64
Logistic regression            Precision: 0.66                    Recall: 0.64
KNN                            Precision: 0.65                    Recall: 0.81

it can be noted that SVM has a high precision i.e. the the predicted values are close to each other.
But KNN has the highest accuracy along with a good precision as well as the highest Recall, which makes it the best model. 

ROC Curve 
Logistic regression   0.62 AUC SCORE
SVM                             0.73 AUC SCORE
KNN                            0.66 AUC SCORE

since the SVM has the highest precision it also indicates that the confidence level of ROC is also the highest among all the model i.e. it is the most suited model rather then picking random TP or TN value for the class
"""
