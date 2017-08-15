import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
datasets = pd.read_csv('Churn_Modelling.csv')
X = datasets.iloc[:, 3:13].values
Y = datasets.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#for gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#splitting

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#making ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#intializing ANN
classifier = Sequential()

#Adding input layer and first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim=11 ))

#second layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting ANN to the training set
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)
#predecting test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


#predecting with the model
new_pred = np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
trans = sc_X.transform(new_pred)
prediction = classifier.predict(trans)
prediction = (prediction > 0.5)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

#evaluating this ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def loop_classifer():
    classifier = Sequential()
#Adding input layer and first hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim=11 ))
#second layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#compiling ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
main_classifier = KerasClassifier(build_fn = loop_classifer, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = main_classifier, X = X_train, y = Y_train, cv = 10)
mean = accuracies.mean()


