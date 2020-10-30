#Start by loading all our lovely models
import numpy
from numpy import isnan
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#load the data from the desktop - this is train_adj in my Github files
url = r'C:\Users\11150049982565796543\Desktop\train_adj.csv'
dataset = read_csv(url, header=0, index_col=0) 

#We'll start by inspecting the data to get a bit of an idea of what it's like
# shape
print(dataset.shape)

# head
set_option('display.width', 100)
print(dataset.head(20))

#Check stats on the different features
set_option('precision', 3)
print(dataset.describe())

#Can also look at how many survived, for example
print(dataset.groupby('Survived').size())

##Visualise the data to get a better idea of what it looks like
# histograms
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()

# density
dataset.plot(kind='density', subplots=True, layout=(2,5), sharex=False, fontsize=1)
pyplot.show()

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,5), sharex=False, sharey=False)
pyplot.show()

# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
FixedFormatter
fig.colorbar(cax)
pyplot.show()

#Combine the above with a pearson's correlation
set_option('precision',2)
print(dataset.corr(method='pearson'))

#My gut feeling is that a SVM model will work best for a binary classification project like this
#but without any better evidence it's safest to try out a few and check their accuracies

##It's time to actually test some models
# Split-out validation dataset
array = dataset.values
X = array[:,0:8].astype(float)
Y = array[:,8]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

#Set up parameters
num_folds = 10
seed = 7
scoring = 'accuracy'

#Organise models
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter = 400)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=None, shuffle=False)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

##Not bad results. Let's look at variance:
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Ok, looks like we got a decent value of 0.8 with LDA. 
# But I'm worried about the 'Tick Adj.'; although I don't understand the basis behind the ticket
# values it seems unlikely that it has anything to do with survivability. Let's remove and try again:
dataset_2 = dataset.drop(['Tick adj'], axis=1)

array = dataset_2.values
X = array[:,0:7].astype(float)
Y = array[:,7]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

#Rerun the models from before...

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=None, shuffle=False)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Interesting! It seems to have significantly improved the accuracy of a number of different models, like LR/LDA
#Although the max model accuracy has not increased, we have a lot more options now, and should continue on this path

##Try see what happens if we standardise data
# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression(solver='lbfgs', max_iter = 400))])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=None, shuffle=False)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# If we try the above without removing the ticket value, there's a slight decrease in our highest SVM value

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#Since SVM and KNN got good results, (81-85%) let's try tune them to see what happens
#First, for KNN we can change the number of n_neighbours

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
neighbors = [1,3,5,7,9,11,13,15,17,19,21]

param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds, random_state=None, shuffle=False)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#Then, for SVM, we can actually tune two different parameters; it's c_values and the kernel_values
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)

model = SVC()
kfold = KFold(n_splits=num_folds, random_state=None, shuffle=False)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#So far the best thing we've accomplished is ~84.7%, using SVM with a C value of 0.5 and using rbf
#But before we go too far, let's try see what happens using ensemble methods

# ensembles
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))

results = []
names = []

for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=None, shuffle=False)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#So these didn't beat my SVM, seems like my gut feeling was right. Let's stick with it

#prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C=0.5, kernel='rbf')
model.fit(rescaledX, Y_train)

# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#Looking at 77%-ish, not as good as the values we were seeing before but still not bad
#Let's apply this to the test sample

#Finally, to apply the model to the test cases
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C=0.5, kernel='rbf')

url = r'C:\Users\11150049982565796543\Desktop\train_adj2.csv' #This is a dataset with the ticket feature already removed
dataset = read_csv(url, header=0, index_col=0)

array = dataset.values
X = array[:,0:7].astype(float)
Y = array[:,7]
fin_model = model.fit(X,Y)

url_test = r'C:\Users\11150049982565796543\Desktop\test_adj.csv'
fin_dataset = read_csv(url_test, header=0, index_col=0) 

fin_array = fin_dataset.values
final_answer = fin_model.predict(fin_array)

numpy.savetxt('values.csv', final_answer, delimiter=",")

#You can then put these values against the original index and submit to Kaggle
#Final result -- 0.77990 

#I also tried some more things to improve results like using SelectKBest and pushing forward
#with the ensemble methods but none of them worked very well, so SVM it'll be for now.